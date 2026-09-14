"""LLMProvider backed by our own vLLM OpenAI-compatible servers.

Drop-in replacement for good_goals.OpenRouterProvider that talks to two local
vLLM servers instead of the OpenRouter API: a chat server (for the goal
proposal / mutation / pairwise-comparison calls) and an embedding server (for
atomic-goal embeddings). Implements the four-method good_goals LLMProvider
protocol (complete / batch_complete / embed / batch_embed) against OpenAI-
compatible endpoints.

Why self-host: GOOD re-sends the whole conversation transcript on every call
and fires ~32 pairwise comparisons per turn that all share that transcript as
a stable leading prefix. On a per-token external API (OpenRouter+Gemini) that
shared prefix isn't discounted -- the concurrent batch races Gemini's implicit
cache and pays full price. vLLM's automatic prefix cache dedups the shared
prefix across the concurrently-scheduled batch, so the cache-friendly prompt
layout finally pays off, and the per-token cost disappears (we already own the
GPUs).

Qwen3 note: Qwen3 ships with "thinking" mode ON by default, which emits a
<think>...</think> block before the answer. GOOD's comparison calls use
max_tokens=10 expecting a bare "1/2/3", so thinking would truncate into
garbage. We disable it per-request via chat_template_kwargs={"enable_thinking":
False}, which vLLM's OpenAI server forwards to the chat template.
"""

from __future__ import annotations  # keeps `list | None` annotations lazy on pre-3.10

import asyncio
import sys

import httpx

# Cap simultaneous in-flight requests so a full batch (up to
# ~max_workers * comparisons_per_round) doesn't exhaust the client-side
# connection pool; vLLM continuous-batches whatever arrives, so this only
# bounds client concurrency, not server throughput.
_MAX_CONCURRENCY = 128


class VLLMProvider:
    def __init__(
        self,
        chat_base_url: str,
        chat_model: str,
        embed_base_url: str,
        embed_model: str,
        timeout: float = 300.0,
        enable_thinking: bool = False,
    ):
        # Normalise to no trailing slash so f"{base}/chat/completions" is clean.
        self.chat_base_url = chat_base_url.rstrip("/")
        self.chat_model = chat_model
        self.embed_base_url = embed_base_url.rstrip("/")
        self.embed_model = embed_model
        self.timeout = timeout
        self.enable_thinking = enable_thinking

        # Failure accounting. THIS EXISTS BECAUSE ITS ABSENCE HID A REAL PROBLEM.
        # `batch_complete`/`batch_embed` gather with return_exceptions=True and map failures to ""
        # so one bad request cannot abort a whole conversation. That is the right resilience
        # choice, but it made a systematic failure *invisible*.
        #
        # What is ESTABLISHED: during the 1k-pool 235B annotation, 377 requests were rejected with
        # "maximum context length is 16384 tokens" (reported message sizes 15k-49.7k). Any of those
        # that arrived through a batch_* call returned "" silently, and the affected turn's goal
        # state is therefore STALE rather than empty, with nothing in the output marking it.
        #
        # What is NOT established: which call type produced those prompts, and which turns were hit.
        # Every measurable component is far too small to explain them -- transcripts max at 8547
        # tokens, individual goal lines at 444 chars, the full 10-set goal distribution at ~11k
        # chars (~3k tokens). Do NOT propagate a mechanism story for this without new evidence; an
        # earlier guess ("transcripts too long", "~2% of turns") was checked and proved wrong.
        # The payload dump in _record_batch_failures is how the next occurrence gets identified.
        self.batch_requests = 0
        self.batch_failures = 0
        self.context_length_failures = 0

    def _chat_payload(self, messages: list[dict], temperature: float, max_tokens: int) -> dict:
        return {
            "model": self.chat_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            # Forwarded to the served model's chat template by vLLM; turns off
            # Qwen3 reasoning so short-max_tokens comparison calls return a bare
            # answer instead of a truncated <think> block.
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

    @staticmethod
    def _is_context_length_error(exc: BaseException) -> bool:
        """Is this a 400 caused by the prompt exceeding the server's --max-model-len?

        Worth singling out: unlike a transient network error, this one is *deterministic* and
        silently degrades output quality for exactly the longest, most goal-rich conversations.
        """
        if isinstance(exc, httpx.HTTPStatusError):
            if exc.response is not None and exc.response.status_code == 400:
                try:
                    return "maximum context length" in exc.response.text
                except Exception:  # noqa: BLE001 - response body may not be readable
                    return False
        return False

    def _record_batch_failures(self, kind: str, results: list, total: int,
                               payloads: list | None = None) -> None:
        """Count failures and, on a context-length rejection, DUMP ENOUGH TO IDENTIFY THE PROMPT.

        The payload dump is not decoration. During the 1k-pool annotation, 377 requests were
        rejected for exceeding 16384 tokens with message sizes reported as 15k-49.7k -- yet every
        component we can measure after the fact is far too small to explain that: transcripts max
        at 8547 tokens, individual goal lines at 444 chars, and the full 10-set goal distribution
        at ~11k chars (~3k tokens). vLLM rejects an over-long request before scheduling, so its log
        never records the prompt content, and the cause remains UNIDENTIFIED. Logging the failing
        payload's size and head here makes the next occurrence self-diagnosing instead of another
        round of inference.
        """
        errors = [(i, r) for i, r in enumerate(results) if isinstance(r, BaseException)]
        self.batch_requests += total
        if not errors:
            return
        ctx_idx = [i for i, e in errors if self._is_context_length_error(e)]
        self.batch_failures += len(errors)
        self.context_length_failures += len(ctx_idx)

        detail = f"{len(errors)}/{total} requests failed"
        if ctx_idx:
            detail += (f"; {len(ctx_idx)} exceeded the server's max context length -- those calls "
                       f"return NOTHING, so the affected turn's goal state is STALE, not empty")
        example = next((repr(e)[:200] for _, e in errors), "")
        print(f"WARNING VLLMProvider.{kind}: {detail}. Substituting empty results. "
              f"first error: {example}", file=sys.stderr, flush=True)

        # Characterise the oversized payloads so the cause is identifiable next time.
        if ctx_idx and payloads:
            for i in ctx_idx[:2]:
                if i >= len(payloads):
                    continue
                p = payloads[i]
                text = ("".join(m.get("content") or "" for m in p)
                        if isinstance(p, list) else str(p))
                print(f"  OVERSIZED PAYLOAD [{kind} idx {i}]: {len(text)} chars "
                      f"(~{len(text)//4} tokens est). head: {text[:300]!r} ... "
                      f"tail: {text[-300:]!r}", file=sys.stderr, flush=True)

    def failure_summary(self) -> dict:
        """Call at the end of a run and LOG IT -- silent zeros are the point of this."""
        return {
            "batch_requests": self.batch_requests,
            "batch_failures": self.batch_failures,
            "context_length_failures": self.context_length_failures,
        }

    def complete(self, messages: list[dict], temperature: float = 0.0, max_tokens: int = 2048) -> str:
        payload = self._chat_payload(messages, temperature, max_tokens)
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.chat_base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()
        return data["choices"][0]["message"]["content"]

    def batch_complete(
        self, messages_list: list[list[dict]], temperature: float = 0.0, max_tokens: int = 2048
    ) -> list[str]:
        async def run_batch():
            sem = asyncio.Semaphore(_MAX_CONCURRENCY)
            limits = httpx.Limits(max_connections=_MAX_CONCURRENCY, max_keepalive_connections=_MAX_CONCURRENCY)
            async with httpx.AsyncClient(timeout=self.timeout, limits=limits) as client:

                async def one(messages):
                    payload = self._chat_payload(messages, temperature, max_tokens)
                    async with sem:
                        response = await client.post(f"{self.chat_base_url}/chat/completions", json=payload)
                        response.raise_for_status()
                        return response.json()["choices"][0]["message"]["content"]

                return await asyncio.gather(*(one(m) for m in messages_list), return_exceptions=True)

        results = asyncio.run(run_batch())
        self._record_batch_failures("batch_complete", results, len(messages_list), messages_list)
        return [r if isinstance(r, str) else "" for r in results]

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.embed_model, "input": text}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.embed_base_url}/embeddings", json=payload)
            response.raise_for_status()
            data = response.json()
        return data["data"][0]["embedding"]

    def batch_embed(self, texts: list[str]) -> list[list[float]]:
        async def run_batch():
            sem = asyncio.Semaphore(_MAX_CONCURRENCY)
            limits = httpx.Limits(max_connections=_MAX_CONCURRENCY, max_keepalive_connections=_MAX_CONCURRENCY)
            async with httpx.AsyncClient(timeout=self.timeout, limits=limits) as client:

                async def one(text):
                    payload = {"model": self.embed_model, "input": text}
                    async with sem:
                        response = await client.post(f"{self.embed_base_url}/embeddings", json=payload)
                        response.raise_for_status()
                        return response.json()["data"][0]["embedding"]

                return await asyncio.gather(*(one(t) for t in texts), return_exceptions=True)

        results = asyncio.run(run_batch())
        self._record_batch_failures("batch_embed", results, len(texts), texts)
        return [r if isinstance(r, list) else [] for r in results]

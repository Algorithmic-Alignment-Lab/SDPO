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

import asyncio

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
        return [r if isinstance(r, list) else [] for r in results]

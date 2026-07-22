"""Precompute GOOD goal-tracking contexts for every (conversation, turn) pair.

GOOD's goal-tracking is pure CPU/network I/O (OpenRouter calls) -- it doesn't
touch a GPU at all. Computing it lazily during training serialized all of a
batch's OpenRouter calls through one Ray actor, which could cost 10+ minutes
of expensive GPU-node time per training step on pure network I/O (see
NOTES.md). Precomputing once, offline, on a CPU-only node lets us parallelize
across conversations instead: each conversation's turns must be walked in
order (GOOD's state evolves turn-by-turn), but different conversations are
fully independent of each other, so a thread pool processes many
conversations concurrently.

Output is a flat {"{conversation_id}:{turn_index}": goal_context_string}
JSON lookup that verl/utils/good_state_cache.py loads directly at training
time -- a pure O(1) dict lookup, no OpenRouter calls, no GOOD dependency at
training time at all.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from good_goals import GOODConfig, GoalState, OpenRouterProvider, format_goals_for_context, update_goals


def _format_conversation_text(messages: list[dict]) -> str:
    """Match GOODChat._format_conversation's convention exactly."""
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def process_conversation(
    conv_id: str, turns: dict[int, list[dict]], provider: OpenRouterProvider, config: GOODConfig
) -> dict[int, str]:
    """Walk one conversation's turns in order, returning {turn_index: goal_context}."""
    results = {}
    state = GoalState()
    for t in sorted(turns):
        conversation_text = _format_conversation_text(turns[t])
        state = update_goals(conversation_text, state, provider, config, update_atomic=True)
        results[t] = format_goals_for_context(state)
    return results


def _atomic_write_json(path: str, obj) -> None:
    """Write JSON via a temp file + rename so a SIGTERM (e.g. Slurm timeout on an
    unattended chunk) can't leave a half-written, unparseable output that would
    break resume. os.replace is atomic on the same filesystem."""
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def _build_provider(args):
    """Construct the LLMProvider GOOD calls for goal inference.

    'openrouter' hits the OpenRouter API (per-token, external). 'vllm' hits our
    own co-located vLLM servers -- a chat server and an embedding server -- so
    GOOD's cache-friendly shared-prefix batches benefit from vLLM's automatic
    prefix caching and we pay GPU-hours we already own instead of per-token.
    """
    if args.provider == "openrouter":
        return OpenRouterProvider(api_key=os.environ["OPENROUTER_API_KEY"])
    if args.provider == "vllm":
        # Imported lazily so the OpenRouter path has no dependency on it.
        from vllm_provider import VLLMProvider

        return VLLMProvider(
            chat_base_url=args.chat_base_url,
            chat_model=args.chat_model,
            embed_base_url=args.embed_base_url,
            embed_model=args.embed_model,
        )
    raise ValueError(f"unknown provider: {args.provider}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conversations_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_workers", type=int, default=16)
    parser.add_argument("--provider", choices=["openrouter", "vllm"], default="openrouter")
    parser.add_argument("--chat_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--chat_model", default="qwen3-32b")
    parser.add_argument("--embed_base_url", default="http://localhost:8001/v1")
    parser.add_argument("--embed_model", default="qwen3-embedding-8b")
    args = parser.parse_args()

    with open(args.conversations_path) as f:
        raw = json.load(f)
    conversations: dict[str, dict[int, list[dict]]] = {
        conv_id: {t["turn_index"]: t["messages"] for t in turns} for conv_id, turns in raw.items()
    }

    provider = _build_provider(args)
    config = GOODConfig()

    # Resume: each conversation is written atomically (all its turns at once) after
    # it finishes, so any conversation_id already present in the output file is fully
    # done and can be skipped. This makes a preempted multi-hour run recoverable --
    # just resubmit against the same --output_path.
    goal_contexts: dict[str, str] = {}
    if os.path.exists(args.output_path):
        with open(args.output_path) as f:
            goal_contexts = json.load(f)
        done_conv_ids = {key.rsplit(":", 1)[0] for key in goal_contexts}
        remaining = {cid: turns for cid, turns in conversations.items() if cid not in done_conv_ids}
        print(
            f"Resuming: {len(done_conv_ids)} conversations already done in {args.output_path}; "
            f"{len(remaining)} of {len(conversations)} remaining.",
            flush=True,
        )
        conversations = remaining

    print(f"Processing {len(conversations)} conversations with {args.max_workers} workers...", flush=True)
    failed_conversations: list[str] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_conversation, conv_id, turns, provider, config): conv_id
            for conv_id, turns in conversations.items()
        }
        done = 0
        for future in as_completed(futures):
            conv_id = futures[future]
            done += 1
            try:
                per_turn = future.result()
            except Exception as e:
                failed_conversations.append(conv_id)
                print(f"[{done}/{len(conversations)}] FAILED conversation={conv_id[:8]}: {e}", flush=True)
                # Write whatever succeeded so far -- a later failure shouldn't lose earlier work.
                _atomic_write_json(args.output_path, goal_contexts)
                continue
            for t, ctx in per_turn.items():
                goal_contexts[f"{conv_id}:{t}"] = ctx
            print(f"[{done}/{len(conversations)}] done conversation={conv_id[:8]} ({len(per_turn)} turns)", flush=True)
            _atomic_write_json(args.output_path, goal_contexts)

    print(f"Wrote {len(goal_contexts)} (conversation, turn) goal contexts to {args.output_path}")
    if failed_conversations:
        print(f"WARNING: {len(failed_conversations)} conversations failed and are missing from the "
              f"output: {failed_conversations}")


if __name__ == "__main__":
    main()

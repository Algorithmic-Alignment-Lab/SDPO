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

from good_goals import (
    GOODConfig,
    GoalState,
    OpenRouterProvider,
    format_goals_for_context,
    get_likely_sets,
    get_plausible_goals,
    update_goals,
)
from good_goals import trace


def _format_conversation_text(messages: list[dict]) -> str:
    """Match GOODChat._format_conversation's convention exactly."""
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def _snapshot_state(state: GoalState) -> dict:
    """Capture the FULL per-turn goal-set distribution that the injected context
    (which only surfaces the top set) throws away: every set with its Beta(alpha,
    beta) and mean/bounds, plus the atomic pool. Uses only public accessors -- no
    change to the library's own behavior."""
    sets = []
    for texts, mean, lower, upper in get_likely_sets(state, top_n=len(state.goal_sets)):
        alpha, beta = state.get_confidence(texts)
        sets.append({
            "goals": texts,
            "mean": mean,
            "lower": lower,
            "upper": upper,
            "alpha": alpha,
            "beta": beta,
        })
    return {
        "num_sets": len(state.goal_sets),
        "focus": sets[0]["goals"] if sets else None,
        "sets": sets,  # sorted by lower bound desc (same ranking as "Current focus")
        "atomic_pool": get_plausible_goals(state),
        "num_atomic": len(state.plausible_state.goals),
        "current_round": state.plausible_state.current_round,
    }


def process_conversation(
    conv_id: str,
    turns: dict[int, list[dict]],
    provider: OpenRouterProvider,
    config: GOODConfig,
    trace_enabled: bool = False,
) -> tuple[dict[int, str], list]:
    """Walk one conversation's turns in order.

    Returns (results, trace_turns) where results is {turn_index: goal_context} and
    trace_turns is a per-turn list of {context, snapshot, events} when tracing is on
    (else None). Trace buffers are thread-local, so concurrent conversations in the
    ThreadPoolExecutor don't interleave.
    """
    results = {}
    trace_turns = [] if trace_enabled else None
    state = GoalState()
    for t in sorted(turns):
        conversation_text = _format_conversation_text(turns[t])
        if trace_enabled:
            trace.begin_turn(conversation_id=conv_id, turn_index=t)
        state = update_goals(conversation_text, state, provider, config, update_atomic=True)
        ctx = format_goals_for_context(state)
        results[t] = ctx
        if trace_enabled:
            trace_turns.append({
                "turn_index": t,
                "context": ctx,
                "snapshot": _snapshot_state(state),
                "events": trace.end_turn(),
            })
    return results, trace_turns


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
        kwargs = {"api_key": os.environ["OPENROUTER_API_KEY"]}
        if args.model:
            kwargs["model"] = args.model
            # Qwen3 (and other hybrid-thinking models) default to a <think> block
            # that empties the max_tokens=10 comparison replies -- turn it off.
            if "qwen" in args.model.lower():
                kwargs["disable_reasoning"] = True
        return OpenRouterProvider(**kwargs)
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
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Split the conversation set across this many independent jobs. Each shard "
        "writes its own --output_path and can run concurrently on its own node; merge the "
        "shard outputs afterwards. Sharding is deterministic (sorted ids, strided).",
    )
    parser.add_argument(
        "--shard_index",
        type=int,
        default=0,
        help="Which shard this job processes (0 <= shard_index < num_shards).",
    )
    parser.add_argument("--provider", choices=["openrouter", "vllm"], default="openrouter")
    parser.add_argument(
        "--model",
        default=None,
        help="Chat model for the OpenRouter provider (e.g. google/gemini-2.5-flash or "
        "qwen/qwen3-32b). Selects the A/B annotation model. Ignored for --provider vllm.",
    )
    parser.add_argument(
        "--proposer",
        choices=["new", "old"],
        default="new",
        help="Fresh goal-set proposal strategy: 'new' = sequential + embedding-seeded "
        "(diverse); 'old' = original concurrent-identical prompts. For A/B comparison.",
    )
    parser.add_argument(
        "--randomize_comparison_order",
        choices=["on", "off"],
        default="on",
        help="R1: randomize Option-1/Option-2 orientation in pairwise set comparisons "
        "to remove the judge's positional bias. 'off' reproduces the pre-R1 fixed "
        "(lower-index = Option 1) ordering for A/B comparison.",
    )
    parser.add_argument(
        "--trace_dir",
        default=None,
        help="If set, enable heavy per-turn diagnostic tracing and write one "
        "trace_{conversation_id}.json per conversation here (full set distribution, "
        "atomic pool, and all generation/comparison/prune events).",
    )
    parser.add_argument("--chat_base_url", default="http://localhost:8000/v1")
    parser.add_argument("--chat_model", default="qwen3-32b")
    parser.add_argument("--embed_base_url", default="http://localhost:8001/v1")
    parser.add_argument("--embed_model", default="qwen3-embedding-8b")
    parser.add_argument(
        "--comparison_leader_weight",
        type=float,
        default=0.0,
        help="Extra pairwise-comparison sampling weight for sets with a high current mean "
        "win-rate, on top of the existing low-evidence preference (see GOODConfig's "
        "docstring). Default 0.0 reproduces today's evidence-only weighting exactly.",
    )
    args = parser.parse_args()

    trace_enabled = args.trace_dir is not None
    if trace_enabled:
        os.makedirs(args.trace_dir, exist_ok=True)
        trace.configure(True)
    model_label = args.model if args.provider == "openrouter" else args.chat_model

    with open(args.conversations_path) as f:
        raw = json.load(f)
    conversations: dict[str, dict[int, list[dict]]] = {
        conv_id: {t["turn_index"]: t["messages"] for t in turns} for conv_id, turns in raw.items()
    }

    # Shard selection happens BEFORE the resume check so each shard resumes against its
    # own output file. Deterministic stride over sorted ids: every conversation lands in
    # exactly one shard regardless of dict ordering.
    if args.num_shards > 1:
        if not 0 <= args.shard_index < args.num_shards:
            raise SystemExit(f"shard_index {args.shard_index} out of range for {args.num_shards} shards")
        all_ids = sorted(conversations)
        mine = set(all_ids[args.shard_index :: args.num_shards])
        conversations = {cid: turns for cid, turns in conversations.items() if cid in mine}
        print(
            f"Shard {args.shard_index}/{args.num_shards}: {len(conversations)} of "
            f"{len(all_ids)} conversations ({sum(len(t) for t in conversations.values())} turns).",
            flush=True,
        )

    provider = _build_provider(args)
    config = GOODConfig()
    config.diverse_fresh_proposals = args.proposer == "new"
    config.randomize_comparison_order = args.randomize_comparison_order == "on"
    config.comparison_leader_weight = args.comparison_leader_weight

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
            executor.submit(process_conversation, conv_id, turns, provider, config, trace_enabled): conv_id
            for conv_id, turns in conversations.items()
        }
        done = 0
        for future in as_completed(futures):
            conv_id = futures[future]
            done += 1
            try:
                per_turn, trace_turns = future.result()
            except Exception as e:
                failed_conversations.append(conv_id)
                print(f"[{done}/{len(conversations)}] FAILED conversation={conv_id[:8]}: {e}", flush=True)
                # Write whatever succeeded so far -- a later failure shouldn't lose earlier work.
                _atomic_write_json(args.output_path, goal_contexts)
                continue
            for t, ctx in per_turn.items():
                goal_contexts[f"{conv_id}:{t}"] = ctx
            if trace_enabled:
                _atomic_write_json(
                    os.path.join(args.trace_dir, f"trace_{conv_id}.json"),
                    {"conversation_id": conv_id, "model": model_label, "turns": trace_turns},
                )
            print(f"[{done}/{len(conversations)}] done conversation={conv_id[:8]} ({len(per_turn)} turns)", flush=True)
            _atomic_write_json(args.output_path, goal_contexts)

    print(f"Wrote {len(goal_contexts)} (conversation, turn) goal contexts to {args.output_path}")
    if failed_conversations:
        print(f"WARNING: {len(failed_conversations)} conversations failed and are missing from the "
              f"output: {failed_conversations}")


if __name__ == "__main__":
    main()

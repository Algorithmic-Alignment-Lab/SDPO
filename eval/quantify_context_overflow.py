"""Enumerate which (conversation, turn) pairs had GOOD prompts too large for the annotation server.

Motivation. The 1k-pool teacher-context regen and the first eval annotation both served the 235B
with `--max-model-len 16384`. GOOD's goal-PROPOSAL call requests `max_tokens=2048`, so any turn
whose prompt exceeds ~14336 tokens was rejected with a 400. Because those calls go through
`VLLMProvider.batch_complete` -- which mapped failures to `""` -- the turn silently received NO new
goal hypotheses while the conversation still reported success. The context is therefore STALE
rather than missing, and nothing in the output marks it.

Why reconstruct instead of parsing logs: vLLM rejects an over-long request at the serving layer
*before* it is scheduled, so the rejected prompts never appear in the log with their content. Only
aggregate counts are recoverable there (377 rejections in the 1k regen; 55 in the first eval run).
Reconstructing from the conversation data is deterministic, complete, and does not depend on log
retention.

Estimate, and its limits. GOOD embeds the conversation as plain `Role: content` lines (confirmed
from a logged request), so transcript tokens dominate. Instruction and goal-list overhead is
modest and configurable via --overhead_tokens. This therefore identifies turns that were
*certainly or very likely* over the limit; it is a lower bound on total affected turns rather than
an exact replay of the annotator.

Usage (in-container; needs transformers):
    python eval/quantify_context_overflow.py \
        --conversations datasets/wildchat_good_1k/conversations.json \
        --label 1k-pool --max_model_len 16384 --completion_tokens 2048 \
        --out eval_runs/context_overflow_1k.json
"""

from __future__ import annotations

import argparse
import json
import statistics


def transcript_text(messages: list) -> str:
    """Plain-text transcript in the shape GOOD sends (confirmed from a logged request)."""
    return "\n".join(f"{m['role'].title()}: {m.get('content') or ''}" for m in messages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-235B-A22B-Instruct-2507-FP8")
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--completion_tokens", type=int, default=2048,
                    help="max_tokens GOOD's proposal call requests.")
    ap.add_argument("--overhead_tokens", type=int, default=250,
                    help="Instruction + goal-list scaffolding around the transcript.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    budget = args.max_model_len - args.completion_tokens - args.overhead_tokens
    print(f"[{args.label}] transcript token budget before rejection: {budget} "
          f"(= {args.max_model_len} - {args.completion_tokens} completion - "
          f"{args.overhead_tokens} overhead)")

    convs = json.load(open(args.conversations))
    affected, all_tokens = [], []
    per_conv = {}
    for cid, turns in convs.items():
        for t in sorted(turns, key=lambda x: x["turn_index"]):
            n = len(tok(transcript_text(t["messages"]), add_special_tokens=False)["input_ids"])
            all_tokens.append(n)
            if n > budget:
                affected.append({"conversation_id": cid, "turn_index": t["turn_index"],
                                 "transcript_tokens": n})
                per_conv.setdefault(cid, []).append(t["turn_index"])

    all_tokens.sort()
    n_turns = len(all_tokens)
    report = {
        "label": args.label,
        "conversations_file": args.conversations,
        "max_model_len": args.max_model_len,
        "completion_tokens": args.completion_tokens,
        "overhead_tokens": args.overhead_tokens,
        "transcript_token_budget": budget,
        "turns_total": n_turns,
        "turns_over_budget": len(affected),
        "turns_over_budget_pct": round(100 * len(affected) / n_turns, 3) if n_turns else 0,
        "conversations_total": len(convs),
        "conversations_with_any_affected_turn": len(per_conv),
        "transcript_tokens": {
            "p50": all_tokens[n_turns // 2] if n_turns else 0,
            "p90": all_tokens[int(n_turns * 0.9)] if n_turns else 0,
            "max": all_tokens[-1] if n_turns else 0,
        },
        "affected_turns": affected,
        "affected_by_conversation": {k: sorted(v) for k, v in sorted(per_conv.items())},
    }
    with open(args.out, "w") as fh:
        json.dump(report, fh, indent=2)

    print(f"[{args.label}] turns {n_turns} in {len(convs)} conversations; "
          f"OVER BUDGET: {len(affected)} turns ({report['turns_over_budget_pct']}%) "
          f"across {len(per_conv)} conversations")
    print(f"[{args.label}] transcript tokens p50={report['transcript_tokens']['p50']} "
          f"p90={report['transcript_tokens']['p90']} max={report['transcript_tokens']['max']}")
    if affected:
        worst = sorted(affected, key=lambda a: -a["transcript_tokens"])[:5]
        print(f"[{args.label}] largest offenders:")
        for a in worst:
            print(f"    {a['conversation_id']}:{a['turn_index']}  {a['transcript_tokens']} tokens")
        deep = [a["turn_index"] for a in affected]
        print(f"[{args.label}] affected turn_index: min={min(deep)} "
              f"median={int(statistics.median(deep))} max={max(deep)} "
              f"-- expect these skewed LATE, since transcripts grow with depth")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

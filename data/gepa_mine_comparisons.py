"""Mine pairwise goal-set comparison instances from GOOD heavy traces.

Each `rank_goal_sets` event in a trace records the (Option-A set, Option-B set) pairs
the judge scored on that turn. This script reconstructs, for every such comparison, the
exact judge input — the conversation transcript at that turn (rebuilt from
conversations.json the same way the precompute driver builds it) plus the two goal sets —
and emits a de-duplicated, stratified pool of instances to calibrate/optimise the judge
prompt against (GEPA teacher labelling happens in gepa_optimize_judge.py; no labels here).

De-dup is orientation-invariant: (setX, setY) and (setY, setX) on the same turn are one
instance, stored in a canonical order so a swapped presentation isn't double counted.

Usage:
  python data/gepa_mine_comparisons.py \
      --conversations_path datasets/wildchat_good_diag30/conversations.json \
      --trace_dirs traces_qwen32b_old traces_qwen32b_new traces_qwen235b_old ... \
      --out datasets/gepa_judge/instances.json \
      --max_per_conv 24 --seed 0
"""

import argparse
import glob
import json
import os
import random


def _format_conversation_text(messages: list[dict]) -> str:
    """Match GOODChat._format_conversation / the precompute driver exactly."""
    return "\n".join(f"{m['role'].capitalize()}: {m['content']}" for m in messages)


def _canon(set_a: list[str], set_b: list[str]) -> tuple[tuple, tuple]:
    """Orientation-invariant key for a pair: order the two sets canonically."""
    ta, tb = tuple(set_a), tuple(set_b)
    return (ta, tb) if ta <= tb else (tb, ta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations_path", required=True)
    ap.add_argument("--trace_dirs", nargs="+", required=True,
                    help="One or more trace_* directories (absolute or cwd-relative).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_per_conv", type=int, default=24,
                    help="Cap unique instances kept per conversation (stratifies over turns).")
    ap.add_argument("--min_transcript_chars", type=int, default=1,
                    help="Skip degenerate empty-transcript turns.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    conversations = json.load(open(args.conversations_path))
    # transcript[conv_id][turn_index] -> formatted transcript string
    transcripts: dict[str, dict[int, str]] = {}
    for cid, entries in conversations.items():
        transcripts[cid] = {
            e["turn_index"]: _format_conversation_text(e["messages"]) for e in entries
        }

    trace_files = []
    for d in args.trace_dirs:
        trace_files += glob.glob(os.path.join(d, "trace_*.json"))
    if not trace_files:
        raise SystemExit(f"no trace_*.json found under {args.trace_dirs}")

    # instances keyed by (conv_id, turn_index, canonical-pair) so the same comparison
    # seen in multiple conditions/turns collapses to one calibration example.
    seen: set[tuple] = set()
    by_conv: dict[str, list[dict]] = {}
    n_comps = 0
    for tf in trace_files:
        tr = json.load(open(tf))
        cid = tr["conversation_id"]
        cid8 = cid[:8]
        for turn in tr["turns"]:
            ti = turn["turn_index"]
            transcript = transcripts.get(cid, {}).get(ti)
            if transcript is None or len(transcript) < args.min_transcript_chars:
                continue
            for ev in turn["events"]:
                if ev.get("kind") != "rank_goal_sets":
                    continue
                for comp in ev.get("comparisons", []):
                    a, b = comp.get("a"), comp.get("b")
                    if not a or not b or a == b:
                        continue
                    n_comps += 1
                    ca, cb = _canon(a, b)
                    key = (cid, ti, ca, cb)
                    if key in seen:
                        continue
                    seen.add(key)
                    by_conv.setdefault(cid8, []).append({
                        "conversation_id": cid,
                        "turn_index": ti,
                        "transcript": transcript,
                        "set_1": list(ca),
                        "set_2": list(cb),
                    })

    # Stratified subsample: cap per conversation, spread across turn depths.
    out = []
    for cid8, insts in sorted(by_conv.items()):
        rng.shuffle(insts)
        # keep a spread over turns: sort by turn then take an even stride up to the cap
        insts.sort(key=lambda x: x["turn_index"])
        if len(insts) > args.max_per_conv:
            step = len(insts) / args.max_per_conv
            insts = [insts[int(i * step)] for i in range(args.max_per_conv)]
        out.extend(insts)

    rng.shuffle(out)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"scanned {len(trace_files)} traces, {n_comps} raw comparisons, "
          f"{len(seen)} unique (conv,turn,pair); kept {len(out)} instances "
          f"across {len(by_conv)} conversations -> {args.out}")


if __name__ == "__main__":
    main()

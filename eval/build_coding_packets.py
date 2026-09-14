"""Build packets for qualitative coding (open coding, then focused coding).

This is deliberately NOT `build_comparison_packets.py`. That builder collapses whitespace
(`" ".join(s.split())`), which is fine for eyeballing prose but destroys exactly what this analysis
needs to see: the bullet structure of the injected goal context, and the markdown/code-block
structure of the responses. Formatting IS a candidate code here, so nothing is reflowed.

Design decisions that matter for the validity of the resulting numbers:

1. **Both arms, blinded.** Each item shows two responses as A/B in a per-item randomised order, with
   no indication of which is the goal-context arm. The same codebook is then applied to both, so the
   output is a *differential* rate (code X in prompted vs in vanilla) rather than a bare rate. A bare
   rate on a corpus the coder has been told is bad measures the coder's willingness to find faults.
2. **Exact model inputs.** The prefix and final user turn come from the row's own
   `prompt_messages`, so the coder reads what the model actually read.
3. **Generous, marked truncation.** Long fields keep head and tail with an explicit elision marker,
   so a coder never mistakes a truncation for a model behaviour (e.g. an unfinished answer).

Modes:
  open     -- stratified sample of items, split across K packets, for inductive code generation.
  focused  -- every item, split across K packets, for applying a fixed codebook.
             `--double N` additionally re-emits N items into extra packets (different packet, so a
             different coder) to make inter-coder agreement measurable rather than assumed.
"""

import argparse
import collections
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalcommon import depth_bucket, length_stratum, load_length_strata  # noqa: E402

ELIDE = "\n\n[... {n} characters elided ...]\n\n"


def stable_bit(*parts: str) -> int:
    """Deterministic coin from the item identity, so A/B order is reproducible across runs."""
    h = hashlib.sha256("|".join(parts).encode()).hexdigest()
    return int(h[:8], 16) & 1


def stable_key(*parts: str) -> int:
    return int(hashlib.sha256("|".join(parts).encode()).hexdigest()[:12], 16)


def keep_ends(s: str, budget: int) -> str:
    """Truncate to `budget` chars keeping head and tail, preserving newlines."""
    s = s or ""
    if len(s) <= budget:
        return s
    head = int(budget * 0.6)
    tail = budget - head
    return s[:head] + ELIDE.format(n=len(s) - budget) + s[-tail:]


def load_rows(path: str, arms: tuple, mode: str) -> dict:
    """(conversation_id, turn_index, arm) -> row, keeping sample 0 only."""
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("arm") not in arms or r.get("mode") != mode:
            continue
        if r.get("sample", 0) != 0:
            continue
        key = (r["conversation_id"], r["turn_index"], r["arm"])
        # Deterministic on duplicates: first occurrence wins, and say so if any appear.
        out.setdefault(key, r)
    return out


def render_item(item_id, conv_id, turn_index, prefix, user_turn, goal_context,
                resp_a, resp_b, args) -> str:
    L = []
    L.append(f"### Item `{item_id}`")
    L.append("")
    L.append(f"- conversation `{conv_id}`, turn index **{turn_index}**")
    L.append("")
    if prefix:
        L.append("<details><summary>Conversation so far (click to expand)</summary>")
        L.append("")
        for m in prefix:
            L.append(f"**{m['role'].upper()}:**")
            L.append("")
            L.append(keep_ends(m.get("content", ""), args.max_prefix_turn))
            L.append("")
        L.append("</details>")
        L.append("")
    else:
        L.append("*(first turn — no prior context)*")
        L.append("")
    L.append("#### Final user turn (what the model must answer)")
    L.append("")
    L.append(keep_ends(user_turn, args.max_user_turn))
    L.append("")
    if goal_context:
        L.append("#### Inferred goal context")
        L.append("")
        L.append("*This block was appended to the final user turn for ONE of the two responses "
                 "below. You are not told which.*")
        L.append("")
        L.append(keep_ends(goal_context, args.max_goal_context))
        L.append("")
    for tag, resp in (("A", resp_a), ("B", resp_b)):
        L.append(f"#### Response {tag}")
        L.append("")
        flags = []
        if resp.get("truncated"):
            flags.append("hit the token limit (truncated)")
        if resp.get("error"):
            flags.append(f"generation error: {resp['error']}")
        if flags:
            L.append(f"> NOTE: this response {'; '.join(flags)}. Do not code that as a model choice.")
            L.append("")
        L.append(keep_ends(resp.get("answer") or "", args.max_answer))
        L.append("")
    L.append("---")
    L.append("")
    return "\n".join(L)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--generations", required=True)
    p.add_argument("--conversations", required=True,
                   help="conversations_subset.json (used only to bound the item universe)")
    p.add_argument("--goal_contexts", default="")
    p.add_argument("--lengths", default="", help="turn_token_lengths.json for the length stratum")
    p.add_argument("--arm_a", default="prompted", help="the arm under study")
    p.add_argument("--arm_b", default="vanilla", help="the reference arm")
    p.add_argument("--mode", default="off")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--stage", choices=("open", "focused"), default="open")
    p.add_argument("--n_items", type=int, default=0,
                   help="open stage: how many items to sample (0 = all)")
    p.add_argument("--packets", type=int, default=6, help="number of packets to split across")
    p.add_argument("--double", type=int, default=0,
                   help="focused stage: re-emit this many items in a second packet for agreement")
    p.add_argument("--seed", default="coding-v1")
    p.add_argument("--max_prefix_turn", type=int, default=3000)
    p.add_argument("--max_user_turn", type=int, default=6000)
    p.add_argument("--max_goal_context", type=int, default=8000)
    p.add_argument("--max_answer", type=int, default=12000)
    args = p.parse_args()

    arms = (args.arm_a, args.arm_b)
    rows = load_rows(args.generations, arms, args.mode)
    goals = json.load(open(args.goal_contexts)) if args.goal_contexts else {}
    lengths = load_length_strata(args.lengths)

    convs = json.load(open(args.conversations))
    universe = [(c, t["turn_index"]) for c, ts in convs.items() for t in ts]

    items, missing = [], collections.Counter()
    for conv_id, ti in universe:
        ra = rows.get((conv_id, ti, args.arm_a))
        rb = rows.get((conv_id, ti, args.arm_b))
        if ra is None or rb is None:
            missing[args.arm_a if ra is None else args.arm_b] += 1
            continue
        items.append((conv_id, ti, ra, rb))
    if missing:
        print(f"WARNING: skipped items with no row: {dict(missing)}", file=sys.stderr)
    if not items:
        sys.exit(f"ERROR: no items with both arms ({args.arm_a}, {args.arm_b}) at mode={args.mode}")

    # Stratify so the sampled items are not all shallow/short; the strata are the same ones the
    # quantitative harness reports, so codes can be cross-tabbed against win rates later.
    def strat(conv_id, ti):
        return (depth_bucket(ti), length_stratum(lengths, conv_id, ti))

    if args.stage == "open" and args.n_items and args.n_items < len(items):
        buckets = collections.defaultdict(list)
        for it in items:
            buckets[strat(it[0], it[1])].append(it)
        for b in buckets.values():
            b.sort(key=lambda it: stable_key(args.seed, it[0], str(it[1])))
        chosen, i = [], 0
        # Round-robin across strata so proportions are roughly preserved without quota arithmetic.
        while len(chosen) < args.n_items:
            added = False
            for k in sorted(buckets):
                if i < len(buckets[k]) and len(chosen) < args.n_items:
                    chosen.append(buckets[k][i])
                    added = True
            if not added:
                break
            i += 1
        items = chosen

    items.sort(key=lambda it: stable_key(args.seed, "order", it[0], str(it[1])))

    os.makedirs(args.out_dir, exist_ok=True)
    packets = collections.defaultdict(list)
    key_rows = []
    for idx, (conv_id, ti, ra, rb) in enumerate(items):
        item_id = f"i{stable_key(args.seed, conv_id, str(ti)):011x}"
        flip = stable_bit(args.seed, "flip", conv_id, str(ti))
        resp_a, resp_b = (rb, ra) if flip else (ra, rb)
        arm_of = {"A": resp_a["arm"], "B": resp_b["arm"]}
        prefix = [m for m in (rb.get("prompt_messages") or [])[:-1]]
        user_turn = (rb.get("prompt_messages") or [{}])[-1].get("content", "")
        gc = goals.get(f"{conv_id}:{ti}", "")
        if isinstance(gc, dict):
            gc = gc.get("goal_context", "")
        body = render_item(item_id, conv_id, ti, prefix, user_turn, gc, resp_a, resp_b, args)
        pk = idx % args.packets
        packets[pk].append((item_id, body))
        d, ln = strat(conv_id, ti)
        key_rows.append({"item_id": item_id, "conversation_id": conv_id, "turn_index": ti,
                         "packet": pk, "A": arm_of["A"], "B": arm_of["B"],
                         "depth": d, "length": ln,
                         "A_truncated": bool(resp_a.get("truncated")),
                         "B_truncated": bool(resp_b.get("truncated"))})

    # Agreement subset: same items, different packet -> a different coder sees them.
    if args.stage == "focused" and args.double:
        dbl = sorted(key_rows, key=lambda r: stable_key(args.seed, "dbl", r["item_id"]))[:args.double]
        by_id = {i: b for pk in packets for i, b in packets[pk]}
        for j, r in enumerate(dbl):
            pk = args.packets + (j // 10)   # ~10 replicate items per extra packet
            packets[pk].append((r["item_id"], by_id[r["item_id"]]))
            key_rows.append({**r, "packet": pk, "agreement_replicate": True})

    manifest = {}
    for pk in sorted(packets):
        path = os.path.join(args.out_dir, f"packet_{pk:03d}.md")
        ids = [i for i, _ in packets[pk]]
        with open(path, "w") as f:
            f.write(f"# Coding packet {pk:03d} ({args.stage} stage)\n\n")
            f.write(f"{len(ids)} items. Responses are blinded: A/B order is randomised per item, "
                    "and the same arm is NOT consistently A.\n\n---\n\n")
            for _, body in packets[pk]:
                f.write(body)
        manifest[os.path.basename(path)] = ids
        print(f"  {os.path.basename(path)}: {len(ids)} items")

    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump({"stage": args.stage, "arm_a": args.arm_a, "arm_b": args.arm_b,
                   "mode": args.mode, "seed": args.seed, "packets": manifest}, f, indent=1)
    with open(os.path.join(args.out_dir, "blinding_key.jsonl"), "w") as f:
        for r in key_rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(packets)} packet(s), {len(key_rows)} rendered item(s) to {args.out_dir}")
    print(f"  blinding key: {args.out_dir}/blinding_key.jsonl  (do NOT show this to coders)")
    strata = collections.Counter((r["depth"], r["length"]) for r in key_rows)
    print("  strata: " + ", ".join(f"{k[0]}/{k[1]}={v}" for k, v in sorted(strata.items())))


if __name__ == "__main__":
    main()

"""Stage 1g: build one blinded per-conversation packet for open-ended subagent review.

This is the qualitative pass that comes BEFORE any rubric exists, and its output is what the
rubric is derived from. So it deliberately asks for observations, not scores: a subagent handed
a scoring scale will produce numbers, and numbers invented before the dimensions are known are
worse than useless.

Extends the pattern proven by data/build_analysis_packets.py on the 9-conversation GOOD focus
check (one compact markdown file per conversation, one subagent each), with three additions the
arm comparison needs:

  * **All arms side by side per turn**, under shuffled neutral tags ("Model A", "Model B", ...)
    re-drawn per packet, so a reviewer cannot learn "C is always the distilled one" across
    packets and cannot carry a prior between conversations.
  * **The injected goal context shown separately** as reference material, clearly marked as what
    the scaffold inferred rather than something the user said.
  * **Thinking traces included when present**, because whether a model's own reasoning
    rediscovers the inferred goals -- with no goal text in its prompt -- is the most direct
    internalization evidence available.

The tag -> arm mapping goes in a separate key file the reviewing subagent is never pointed at.

Usage:
    python eval/build_comparison_packets.py \
        --generations eval_runs/generations.jsonl \
        --conversations datasets/wildchat_eval_250/conversations.json \
        --goal_contexts datasets/wildchat_eval_250/goal_contexts_235b.json \
        --n 30 --mode off --out_dir eval_runs/packets/off
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evalcommon import (  # noqa: E402
    format_transcript,
    item_strata,
    load_jsonl,
    load_length_strata,
    stable_shuffle,
    stratified_sample,
)

TAGS = ["Model A", "Model B", "Model C", "Model D", "Model E", "Model F", "Model G"]

REVIEW_PROMPT = """\
You are characterising HOW several model responses differ, for a research evaluation.

**You are not judging quality.** Do not say which response is better, do not rank them, do not
score anything, and do not recommend one. A reviewer who reports "C is the best" has produced
nothing usable here. What we want is a precise description of *interesting differences* -- and
whether those differences have anything to do with what the user is actually trying to accomplish.

Background. Each packet is one real multi-turn conversation. At several turn boundaries, several
different models each produced the next assistant reply. Some models were given an extra block of
"inferred user goals" in their prompt; others were given only the raw conversation. You are NOT
told which is which and should not guess -- describe behaviour, never labels.

For each turn, and then for the packet overall:

1. **What actually differs.** Quote short fragments. Be specific: different content, different
   ordering, different assumptions, different scope, a question asked instead of an answer given.
   If the responses are substantively the same, say so plainly -- "no interesting difference" is a
   real and valuable finding, and manufacturing differences that are not there is the main failure
   mode of this task.
2. **Does the difference track the user's goals?** The user has some underlying aim, which may be
   only partly stated. Does any response engage something the user evidently wants but did not
   say? Does any response pursue something the user did not ask for? An inferred-goals block is
   shown for reference -- treat it as one hypothesis about the user's aims, NOT as ground truth,
   and note where you disagree with it.
3. **Substance or form?** For every difference you flag, say whether it changes what the response
   *does* or only how it *sounds* (length, structure, confidence, formatting). This distinction is
   the most important thing this review produces, because a difference in form alone is easy to
   mistake for a difference in capability.
4. **Anything surprising.** Behaviour you did not expect, in either direction -- including a
   response that seems to understand the user unusually well, and a response that breaks down,
   repeats itself, drifts off-topic, or answers a different question than the one asked. If a
   response degenerates, say where it starts and what it does instead.
5. **Reasoning traces**, where included: what does the trace reason about, and does that reasoning
   show up in the response?

End with **the 3-5 dimensions along which these responses most differ**, phrased neutrally (as
axes of variation, not as good/bad). If they barely differ, say that instead.
"""


def _trunc(s: str, n: int) -> str:
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + " …[truncated]"


def build_packet(conv_id: str, turns_data: list, tag_map: dict, max_answer: int,
                 max_think: int) -> str:
    arm_by_tag = {v: k for k, v in tag_map.items()}
    lines = [
        f"# Conversation {conv_id}",
        "",
        "Each turn below shows the conversation up to that point, the goals a goal-inference "
        "system had inferred at that point (reference only -- the user did not say these), and "
        "the replies several models produced. Model tags are arbitrary and are re-drawn for "
        "every packet; they carry no meaning across packets.",
        "",
    ]

    for td in turns_data:
        strata_note = ", ".join(
            f"{k}={v}" for k, v in (td.get("strata") or {}).items() if k != "defect_labels")
        lines += [
            "---",
            "",
            f"## Turn {td['turn_index']}" + (f"  _(strata: {strata_note})_" if strata_note else ""),
            "",
            "### Conversation so far",
            "",
            td["transcript"],
            "",
        ]

        if td.get("goal_context"):
            lines += [
                "### Inferred user goals at this turn (reference; NOT said by the user)",
                "",
                "```",
                _trunc(td["goal_context"], 4000),
                "```",
                "",
            ]

        lines += ["### Model replies", ""]
        # Tags appear in fixed alphabetical order within a packet so the reader is not also
        # tracking a shifting layout; what is randomised is the tag -> arm mapping.
        for tag in sorted(arm_by_tag):
            arm = arm_by_tag[tag]
            resp = td["responses"].get(arm)
            if resp is None:
                continue
            lines += [f"**{tag}:**", "", _trunc(resp["answer"], max_answer), ""]
            if (resp.get("think") or "").strip():
                lines += [
                    f"<details><summary>{tag} reasoning trace</summary>", "",
                    "```", _trunc(resp["think"], max_think), "```", "", "</details>", "",
                ]
            if resp.get("truncated"):
                lines += [f"_(note: {tag}'s reply hit the token limit and is cut off)_", ""]

    lines += ["---", "", "## What to report", "", REVIEW_PROMPT]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--goal_contexts", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--mode", default="off", choices=["off", "on"])
    ap.add_argument("--arms", default="vanilla,prompted,distilled-fw,distilled-lora")
    ap.add_argument("--n", type=int, default=30, help="Conversations to sample.")
    ap.add_argument("--max_turns_per_packet", type=int, default=4)
    ap.add_argument("--max_answer_chars", type=int, default=2500)
    ap.add_argument("--max_think_chars", type=int, default=3000)
    ap.add_argument("--length_strata", default=None)
    ap.add_argument("--defect_labels", default=None)
    ap.add_argument("--strata", default="depth,length")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    conversations = json.load(open(args.conversations))
    contexts = json.load(open(args.goal_contexts))
    lengths = load_length_strata(args.length_strata)
    defects = json.load(open(args.defect_labels)) if args.defect_labels else None

    gens = {}
    for r in load_jsonl(args.generations):
        if r.get("error") or r["mode"] != args.mode or r["sample"] != 0:
            continue
        gens[(r["conversation_id"], r["turn_index"], r["arm"])] = r

    # Only turns where EVERY requested arm produced a response, so a packet never shows a
    # partial comparison that a reviewer would read as a difference between models.
    candidates = []
    for conv_id, turns in conversations.items():
        usable = [t for t in sorted(turns, key=lambda x: x["turn_index"])
                  if all((conv_id, t["turn_index"], a) in gens for a in arms)]
        if not usable:
            continue
        cand = {"conversation_id": conv_id, "turn_index": usable[0]["turn_index"],
                "usable": usable}
        cand["strata"] = item_strata(cand, lengths, defects)
        candidates.append(cand)

    if not candidates:
        sys.exit(f"no conversations have all arms {arms} generated for mode={args.mode}")

    picked = stratified_sample(candidates, args.n, tuple(args.strata.split(",")),
                              salt=f"packets|{args.mode}")
    os.makedirs(args.out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(args.out_dir, "packet_*.md")):
        os.remove(old)

    key_rows = []
    for cand in picked:
        conv_id = cand["conversation_id"]
        shuffled = stable_shuffle(list(arms), "tags", conv_id, args.mode)
        tag_map = {arm: TAGS[i] for i, arm in enumerate(shuffled)}

        # Spread the shown turns across the conversation rather than taking the first few, so a
        # packet covers late goal state (where tracking should matter most) and not only the
        # opening.
        usable = cand["usable"]
        if len(usable) > args.max_turns_per_packet:
            step = len(usable) / args.max_turns_per_packet
            usable = [usable[int(i * step)] for i in range(args.max_turns_per_packet)]

        turns_data = []
        for t in usable:
            ti = t["turn_index"]
            responses = {}
            for arm in arms:
                row = gens[(conv_id, ti, arm)]
                responses[arm] = {"answer": row.get("answer") or "",
                                  "think": row.get("think") or "",
                                  "truncated": bool(row.get("truncated"))}
            turns_data.append({
                "turn_index": ti,
                "transcript": format_transcript(t["messages"]),
                "goal_context": contexts.get(f"{conv_id}:{ti}", ""),
                "responses": responses,
                "strata": item_strata({"conversation_id": conv_id, "turn_index": ti},
                                      lengths, defects),
            })

        md = build_packet(conv_id, turns_data, tag_map, args.max_answer_chars,
                          args.max_think_chars)
        path = os.path.join(args.out_dir, f"packet_{conv_id}.md")

        # Sanity: no arm name may appear in what a reviewer reads. Checked before writing.
        leaked = [a for a in arms if a in md]
        if leaked:
            sys.exit(f"BLINDING FAILURE: arm name(s) {leaked} would appear in {path}")

        with open(path, "w") as fh:
            fh.write(md)
        key_rows.append({"packet": os.path.basename(path), "conversation_id": conv_id,
                         "mode": args.mode, "tag_map": tag_map,
                         "turns": [td["turn_index"] for td in turns_data],
                         "strata": cand["strata"]})

    with open(os.path.join(args.out_dir, "packet_key.jsonl"), "w") as fh:
        for row in key_rows:
            fh.write(json.dumps(row) + "\n")

    print(f"wrote {len(key_rows)} packets to {args.out_dir} (mode={args.mode}, arms={arms})")
    print(f"  tag->arm key: {args.out_dir}/packet_key.jsonl  (do NOT show this to reviewers)")
    print("  dispatch one subagent per packet; each reports observations, not scores")


if __name__ == "__main__":
    main()

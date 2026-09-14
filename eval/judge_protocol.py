"""Prepare blinded judging batches for Claude Code subagents, and aggregate their verdicts.

Design constraints that the plan makes non-negotiable, all enforced here in code rather than
left to whoever writes the subagent prompt:

* **Blinding.** Batch files contain no arm names, no model names, and no ordering hint. The
  mapping from item -> arm lives in a separate key file the judging subagent is never pointed
  at. Aggregation un-blinds.
* **Both presentation orders.** Every sampled (pair, item) is judged twice, once with each
  response first. The order-flip disagreement rate is then a measured property reported next to
  every win rate -- an unreported position bias is not a measurement.
* **What each judge may see.** The QUALITY judge sees the conversation and two answers, and
  never the goal context (which would bias it toward whichever arm echoes its wording) and never
  the `<think>` trace (reasoning verbosity is not response quality). The COVERAGE judge is the
  only one shown the goal context, as a reference. These are different tasks precisely because
  they must see different things.
* **Ties are available and unparseable verdicts are NOT defaulted.** `gepa_common.parse_choice`
  defaults to 3 (tie) when nothing parses, which is correct for keeping GOOD's algorithm running
  but wrong here: silently converting a failed judgement into "no difference" biases every
  comparison toward the null we are testing against. Unparseable verdicts are counted and
  excluded.

Usage:
    # prepare
    python eval/judge_protocol.py prepare-quality \
        --generations eval_runs/generations.jsonl \
        --pairs distilled-fw:vanilla,distilled-fw:prompted,prompted:prompted-placebo \
        --n 300 --modes off,on --out_dir eval_runs/judge/quality
    python eval/judge_protocol.py prepare-quality \
        --generations eval_runs/generations.jsonl --pairs vanilla:vanilla \
        --n 100 --modes off,on --out_dir eval_runs/judge/control     # negative control

    # ... subagents judge each batch file, appending verdicts to <batch>.verdicts.jsonl ...

    python eval/judge_protocol.py aggregate-quality --batch_dir eval_runs/judge/quality
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import item_rendering  # noqa: E402
from evalcommon import (  # noqa: E402
    bootstrap_ci_by_conversation,
    format_transcript,
    item_strata,
    load_jsonl,
    load_length_strata,
    stable_int,
    stratified_sample,
)

QUALITY_INSTRUCTIONS = """\
You are judging which of two assistant responses is better, for research evaluation.

For each item you are given the conversation so far and two candidate responses to the final
user turn. Decide which response is better *as a reply to that final user turn*, considering
whether it addresses what the user actually wants (including anything implied but not stated),
is helpful and specific, and avoids inventing constraints the user never expressed.

Judge only the two responses in front of you. Do not reward length, formatting, or confident
tone for their own sake -- a shorter reply that serves the user better is better.

Return, for EACH item, one JSON object per line with these fields:
  {"item_id": "<the item's id, copied exactly>",
   "choice": <1, 2, 3, or 4>,
   "reason": "<one sentence, max ~25 words>"}

choice:
  1 = Response 1 is better
  2 = Response 2 is better
  3 = they are of equally good quality
  4 = both are bad

Use 3 and 4 genuinely -- do not force a winner when there isn't one, and do not avoid them.
Output ONLY those JSON lines, one per item, nothing else.
"""

COVERAGE_INSTRUCTIONS = """\
You are rating how well an assistant response engages a set of inferred user goals.

For each item you are given the conversation so far, a REFERENCE list of goals that a
goal-inference system inferred for the user at this point, and ONE candidate response.

Rate how well the response engages those inferred goals. This is not a quality judgement: a
response can be well written and still ignore the goals, or plain and still address them well.
Do NOT reward the response for restating or quoting the goal list -- what matters is whether its
substance reflects them.

Return, for EACH item, one JSON object per line:
  {"item_id": "<copied exactly>",
   "coverage": <0, 1, 2, or 3>,
   "reason": "<one sentence, max ~25 words>"}

coverage:
  0 = ignores or contradicts the inferred goals
  1 = touches on them incidentally
  2 = clearly engages the main inferred goal(s)
  3 = engages the main goal(s) and respects the stated concerns/constraints

Output ONLY those JSON lines, one per item, nothing else.
"""

TRACE_INSTRUCTIONS = """\
You are analysing an assistant's internal reasoning trace, for research evaluation.

For each item you are given the conversation so far and the assistant's reasoning trace (its
private thinking before answering). The trace was produced with NO goal information in the
prompt -- whatever it reasons about, it inferred itself.

Return, for EACH item, one JSON object per line:
  {"item_id": "<copied exactly>",
   "reasons_about_goals": true|false,
   "reasons_about_ambiguity": true|false,
   "inferred_goals": ["<short paraphrase>", ...],
   "reason": "<one sentence, max ~25 words>"}

  reasons_about_goals     : does the trace explicitly consider what the user is trying to
                            achieve, beyond restating the literal request?
  reasons_about_ambiguity : does it note something under-specified, ambiguous, or a choice the
                            user has not made?
  inferred_goals          : short paraphrases of any user goals/preferences/constraints the
                            trace identifies. [] if none. Paraphrase what the TRACE says, do not
                            add goals of your own.

Output ONLY those JSON lines, one per item, nothing else.
"""


def index_generations(rows: list[dict]) -> dict:
    """(conversation_id, turn_index, arm, mode, sample) -> row, errors dropped."""
    out = {}
    for r in rows:
        if r.get("error"):
            continue
        out[(r["conversation_id"], r["turn_index"], r["arm"], r["mode"], r["sample"])] = r
    return out


def write_batches(items: list[dict], instructions: str, out_dir: str, batch_size: int,
                  kind: str, key_rows: list[dict]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "batch_*.json")):
        os.remove(old)

    n_batches = 0
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        path = os.path.join(out_dir, f"batch_{start // batch_size:04d}.json")
        with open(path, "w") as fh:
            json.dump({
                "kind": kind,
                "instructions": instructions,
                "verdict_file": path + ".verdicts.jsonl",
                "n_items": len(chunk),
                "items": chunk,
            }, fh, indent=2, ensure_ascii=False)
        n_batches += 1

    # The key is what un-blinds the verdicts. Written OUTSIDE the batch files so a judging
    # subagent pointed at a batch cannot see arm identities even accidentally.
    with open(os.path.join(out_dir, "blinding_key.jsonl"), "w") as fh:
        for row in key_rows:
            fh.write(json.dumps(row) + "\n")

    print(f"wrote {n_batches} batch file(s) ({len(items)} judging items) to {out_dir}")
    print(f"  blinding key: {out_dir}/blinding_key.jsonl  (do NOT show this to judges)")


def cmd_prepare_quality(args):
    gens = index_generations(load_jsonl(args.generations))
    lengths = load_length_strata(args.length_strata)
    defects = json.load(open(args.defect_labels)) if args.defect_labels else None

    pairs = [tuple(p.split(":")) for p in args.pairs.split(",") if p]
    for p in pairs:
        if len(p) != 2:
            sys.exit(f"--pairs entries must be arm_a:arm_b, got {p}")
    modes = [m for m in args.modes.split(",") if m]

    items, key_rows = [], []
    for arm_a, arm_b in pairs:
        for mode in modes:
            # The negative control compares an arm with ITSELF at two different seeds, which is
            # the only case where both sides come from the same arm.
            same_arm = arm_a == arm_b
            sample_a, sample_b = (0, 1) if same_arm else (0, 0)

            candidates = []
            for (conv, turn, arm, m, samp), row in gens.items():
                if arm != arm_a or m != mode or samp != sample_a:
                    continue
                other = gens.get((conv, turn, arm_b, mode, sample_b))
                if other is None:
                    continue
                if not (row.get("answer") or "").strip() or not (other.get("answer") or "").strip():
                    continue  # an empty answer is not judgeable; counted by the generator's summary
                cand = {"conversation_id": conv, "turn_index": turn, "row_a": row, "row_b": other}
                cand["strata"] = item_strata(cand, lengths, defects)
                candidates.append(cand)

            if not candidates:
                print(f"  WARNING: no judgeable items for {arm_a} vs {arm_b} [{mode}]")
                continue

            picked = stratified_sample(candidates, args.n, tuple(args.strata.split(",")),
                                       salt=f"quality|{arm_a}|{arm_b}|{mode}")
            print(f"  {arm_a} vs {arm_b} [{mode}]: {len(picked)} items "
                  f"(from {len(candidates)} available) x 2 orders")

            for cand in picked:
                base = f"{cand['conversation_id']}:{cand['turn_index']}"
                for order in (0, 1):
                    # order 0 -> Response 1 is arm_a; order 1 -> Response 1 is arm_b.
                    first, second = ((cand["row_a"], cand["row_b"]) if order == 0
                                     else (cand["row_b"], cand["row_a"]))
                    item_id = f"q{stable_int('q', arm_a, arm_b, mode, base, order):08x}"
                    items.append({
                        "item_id": item_id,
                        "conversation": format_transcript(
                            _judged_prefix(first) + [_final_user_turn(first)]),
                        "response_1": (first.get("answer") or "").strip(),
                        "response_2": (second.get("answer") or "").strip(),
                    })
                    key_rows.append({
                        "item_id": item_id, "kind": "quality",
                        "conversation_id": cand["conversation_id"],
                        "turn_index": cand["turn_index"], "mode": mode,
                        "pair": f"{arm_a}|{arm_b}", "order": order,
                        "arm_response_1": first["arm"], "arm_response_2": second["arm"],
                        "sample_response_1": first["sample"], "sample_response_2": second["sample"],
                        "strata": cand["strata"],
                    })

    # Mandatory leak scan: a "blind" item whose transcript carries any goal-context text is
    # not a measurement (SHARD0_FINDINGS.md §1). Fail loudly and write nothing.
    dirty = [(it["item_id"], reasons) for it in items
             if (reasons := item_rendering.scan_text_for_leaks(it["conversation"]))]
    if dirty:
        sys.exit(f"LEAK SCAN FAILED on {len(dirty)}/{len(items)} quality items "
                 f"(first: {dirty[0]}) -- refusing to write batches.")

    # SEPARATE THE TWO ORDERS INTO DISJOINT HALVES OF THE BATCH SEQUENCE.
    #
    # Shuffling alone was not enough, and a judge caught it: it noticed both presentation orders of
    # the same pairing inside its own batch and reported that its verdicts "came out consistent" on
    # them. That destroys the flip rate as a position-bias check -- a judge that RECOGNISES the
    # duplicate answers consistently for that reason, not because it is position-invariant, so the
    # bias looks smaller than it is. Measured on the first run: 7% of pairings shared a batch, and
    # 17% were seen by the same judge once batches were handed out several per agent.
    #
    # Emitting all order-0 items first and all order-1 items second guarantees the two orders land
    # in different batch files. Callers who hand several batches to one judge should take them from
    # opposite ends of the sequence, or accept that only cross-half pairings measure flips cleanly.
    by_order = {0: [], 1: []}
    for it, k in zip(items, key_rows):
        by_order[k["order"]].append(it)
    for o in (0, 1):
        by_order[o] = sorted(by_order[o], key=lambda it: stable_int("shuf", it["item_id"]))
    items = by_order[0] + by_order[1]
    write_batches(items, QUALITY_INSTRUCTIONS, args.out_dir, args.batch_size, "quality", key_rows)


# Context stripping moved to item_rendering.py (2026-09-05) so the coding-study packet
# builders and this harness cannot drift apart again -- the Sep-4 pairwise leak
# (SHARD0_FINDINGS.md §1) happened in a builder that reimplemented what these functions
# already did correctly. The full WHY-THIS-MATTERS history lives in that module's docstrings.
_judged_prefix = item_rendering.judged_prefix
_final_user_turn = item_rendering.final_user_turn


def cmd_prepare_coverage(args):
    gens = index_generations(load_jsonl(args.generations))
    lengths = load_length_strata(args.length_strata)
    defects = json.load(open(args.defect_labels)) if args.defect_labels else None
    contexts = json.load(open(args.goal_contexts))
    arms = [a for a in args.arms.split(",") if a]
    modes = [m for m in args.modes.split(",") if m]

    items, key_rows = [], []
    for mode in modes:
        # Sample the TURNS once per mode, then rate every arm on those same turns, so coverage is
        # comparable across arms rather than each arm being scored on a different subset.
        turn_pool = {}
        for (conv, turn, arm, m, samp), row in gens.items():
            if m != mode or samp != 0 or arm not in arms:
                continue
            turn_pool.setdefault((conv, turn), {})[arm] = row
        full = [{"conversation_id": c, "turn_index": t, "rows": r}
                for (c, t), r in turn_pool.items() if all(a in r for a in arms)
                and contexts.get(f"{c}:{t}", "").strip()]
        for cand in full:
            cand["strata"] = item_strata(cand, lengths, defects)
        if not full:
            print(f"  WARNING: no coverage-ratable turns for [{mode}]")
            continue
        picked = stratified_sample(full, args.n, tuple(args.strata.split(",")),
                                   salt=f"coverage|{mode}")
        print(f"  coverage [{mode}]: {len(picked)} turns x {len(arms)} arms")

        for cand in picked:
            ctx = contexts[f"{cand['conversation_id']}:{cand['turn_index']}"]
            for arm in arms:
                row = cand["rows"][arm]
                if not (row.get("answer") or "").strip():
                    continue
                item_id = f"c{stable_int('c', arm, mode, cand['conversation_id'], cand['turn_index']):08x}"
                items.append({
                    "item_id": item_id,
                    "conversation": format_transcript(
                        _judged_prefix(row) + [_final_user_turn(row)]),
                    "reference_goals": ctx,
                    "response": (row.get("answer") or "").strip(),
                })
                key_rows.append({
                    "item_id": item_id, "kind": "coverage",
                    "conversation_id": cand["conversation_id"],
                    "turn_index": cand["turn_index"], "mode": mode, "arm": arm,
                    "strata": cand["strata"],
                })

    items = sorted(items, key=lambda it: stable_int("shuf", it["item_id"]))
    write_batches(items, COVERAGE_INSTRUCTIONS, args.out_dir, args.batch_size, "coverage", key_rows)


def cmd_prepare_trace(args):
    gens = index_generations(load_jsonl(args.generations))
    lengths = load_length_strata(args.length_strata)
    arms = [a for a in args.arms.split(",") if a]

    items, key_rows = [], []
    turn_pool = {}
    for (conv, turn, arm, m, samp), row in gens.items():
        # Thinking-on only: there is no trace to analyse with thinking off.
        if m != "on" or samp != 0 or arm not in arms:
            continue
        if not (row.get("think") or "").strip():
            continue
        turn_pool.setdefault((conv, turn), {})[arm] = row

    full = [{"conversation_id": c, "turn_index": t, "rows": r}
            for (c, t), r in turn_pool.items() if all(a in r for a in arms)]
    for cand in full:
        cand["strata"] = item_strata(cand, lengths, None)
    if not full:
        sys.exit("no thinking-on traces found for all requested arms")
    picked = stratified_sample(full, args.n, tuple(args.strata.split(",")), salt="trace")
    print(f"  trace: {len(picked)} turns x {len(arms)} arms")

    for cand in picked:
        for arm in arms:
            row = cand["rows"][arm]
            item_id = f"t{stable_int('t', arm, cand['conversation_id'], cand['turn_index']):08x}"
            items.append({
                "item_id": item_id,
                "conversation": format_transcript(
                    _judged_prefix(row) + [_final_user_turn(row)]),
                "reasoning_trace": (row.get("think") or "").strip(),
            })
            key_rows.append({
                "item_id": item_id, "kind": "trace",
                "conversation_id": cand["conversation_id"],
                "turn_index": cand["turn_index"], "mode": "on", "arm": arm,
                "strata": cand["strata"],
            })

    items = sorted(items, key=lambda it: stable_int("shuf", it["item_id"]))
    write_batches(items, TRACE_INSTRUCTIONS, args.out_dir, args.batch_size, "trace", key_rows)


def load_verdicts(batch_dir: str) -> tuple[dict, int, int]:
    """Read every <batch>.verdicts.jsonl. Returns (by_item_id, n_unparseable, n_missing_id)."""
    verdicts, bad, no_id = {}, 0, 0
    for path in sorted(glob.glob(os.path.join(batch_dir, "batch_*.json.verdicts.jsonl"))):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                v = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not v.get("item_id"):
                no_id += 1
                continue
            verdicts[v["item_id"]] = v
    return verdicts, bad, no_id


def cmd_aggregate_quality(args):
    key = {r["item_id"]: r for r in load_jsonl(os.path.join(args.batch_dir, "blinding_key.jsonl"))}
    verdicts, n_bad, n_no_id = load_verdicts(args.batch_dir)

    # Which judge saw each item? Batch files are handed out --batches_per_judge at a time, so the
    # judge index is the batch index divided by that. Needed to tell whether a pairing's two orders
    # were judged independently (see the flip-rate note below).
    judge_of = {}
    explicit = {}
    if getattr(args, "judge_batches", ""):
        for jid, grp in enumerate(args.judge_batches.split(";")):
            for bi in (x.strip() for x in grp.split(",") if x.strip()):
                explicit[int(bi)] = jid
    for bi, bf in enumerate(sorted(glob.glob(os.path.join(args.batch_dir, "batch_*.json")))):
        jid = explicit.get(bi, bi // max(1, args.batches_per_judge))
        for it in json.load(open(bf))["items"]:
            judge_of[it["item_id"]] = jid
    print(f"verdicts: {len(verdicts)} parsed, {n_bad} unparseable, {n_no_id} missing item_id, "
          f"of {len(key)} prepared items")
    if n_bad or n_no_id:
        print("  NOTE: unparseable verdicts are EXCLUDED, never defaulted to a tie -- silently "
              "converting a failed judgement into 'no difference' would bias toward the null.")

    # (pair, mode, conv, turn) -> {order: outcome for arm_a}, outcome in {"a","b","tie","bad"}
    grouped: dict[tuple, dict] = {}
    for item_id, k in key.items():
        v = verdicts.get(item_id)
        if v is None:
            continue
        choice = v.get("choice")
        if choice not in (1, 2, 3, 4):
            continue
        arm_a = k["pair"].split("|")[0]
        if choice in (3, 4):
            outcome = "tie"
        else:
            winner = k["arm_response_1"] if choice == 1 else k["arm_response_2"]
            # Self-comparison (negative control): both arms are equal, so resolve by which
            # SAMPLE won rather than by arm name, which would be ambiguous.
            if k["arm_response_1"] == k["arm_response_2"]:
                won_sample = (k["sample_response_1"] if choice == 1 else k["sample_response_2"])
                outcome = "a" if won_sample == 0 else "b"
            else:
                outcome = "a" if winner == arm_a else "b"
        gk = (k["pair"], k["mode"], k["conversation_id"], k["turn_index"])
        cell = grouped.setdefault(gk, {"strata": k["strata"], "judges": {}})
        cell[k["order"]] = outcome
        cell["judges"][k["order"]] = judge_of.get(item_id)

    results = {}
    for (pair, mode) in sorted({(gk[0], gk[1]) for gk in grouped}):
        cells = {gk: v for gk, v in grouped.items() if gk[0] == pair and gk[1] == mode}
        both = {gk: v for gk, v in cells.items() if 0 in v and 1 in v}
        flips = sum(1 for v in both.values()
                    if {v[0], v[1]} == {"a", "b"})
        # Flip rate is only a genuine position-bias check when the two orders were judged
        # INDEPENDENTLY. Where one judge saw both, it may have recognised the duplicate and
        # answered consistently for that reason, which understates bias. Reported separately.
        indep = {gk: v for gk, v in both.items()
                 if v["judges"].get(0) is None or v["judges"].get(0) != v["judges"].get(1)}
        flips_indep = sum(1 for v in indep.values() if {v[0], v[1]} == {"a", "b"})
        # Score every judgement (both orders) as a win for arm_a: 1 win, 0.5 tie, 0 loss.
        by_conv: dict[str, list[float]] = {}
        ties = total = 0
        for gk, v in cells.items():
            for order in (0, 1):
                if order not in v:
                    continue
                o = v[order]
                by_conv.setdefault(gk[2], []).append(1.0 if o == "a" else 0.5 if o == "tie" else 0.0)
                ties += int(o == "tie")
                total += 1
        point, lo, hi = bootstrap_ci_by_conversation(by_conv, n_boot=args.n_boot,
                                                     salt=f"{pair}|{mode}")
        arm_a, arm_b = pair.split("|")
        res = {
            "pair": pair, "mode": mode, "arm_a": arm_a, "arm_b": arm_b,
            "judgements": total, "items_both_orders": len(both),
            "conversations": len(by_conv),
            "win_rate_a": round(point, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "tie_fraction": round(ties / total, 4) if total else None,
            "order_flip_disagreement": round(flips / len(both), 4) if both else None,
            "order_flip_disagreement_independent": round(flips_indep / len(indep), 4) if indep else None,
            "pairings_independently_judged": len(indep),
            "pairings_same_judge": len(both) - len(indep),
        }

        # Per-stratum breakdown, reported alongside every headline number.
        strata_out = {}
        for skey in ("depth", "length", "defect"):
            buckets: dict[str, dict[str, list[float]]] = {}
            for gk, v in cells.items():
                label = (v.get("strata") or {}).get(skey)
                if label is None:
                    continue
                for order in (0, 1):
                    if order not in v:
                        continue
                    o = v[order]
                    buckets.setdefault(label, {}).setdefault(gk[2], []).append(
                        1.0 if o == "a" else 0.5 if o == "tie" else 0.0)
            if buckets:
                strata_out[skey] = {}
                for label, bc in sorted(buckets.items()):
                    p, l, h = bootstrap_ci_by_conversation(bc, n_boot=args.n_boot,
                                                           salt=f"{pair}|{mode}|{skey}|{label}")
                    strata_out[skey][label] = {
                        "n_judgements": sum(len(x) for x in bc.values()),
                        "win_rate_a": round(p, 4), "ci95": [round(l, 4), round(h, 4)]}
        res["strata"] = strata_out
        results[f"{pair}|{mode}"] = res

    print(f"\n{'pair':<38} {'mode':<5} {'n':>6} {'winA':>7} {'ci95':>17} {'tie%':>6} {'flip%':>6}")
    for k, r in results.items():
        print(f"{r['arm_a']+' vs '+r['arm_b']:<38} {r['mode']:<5} {r['judgements']:>6} "
              f"{r['win_rate_a']:>7.3f} "
              f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]".rjust(18) +
              f" {100*(r['tie_fraction'] or 0):>5.1f} {100*(r['order_flip_disagreement'] or 0):>5.1f}")

    print("\nwinA is the win rate of the FIRST-named arm, ties counted as 0.5. CIs are a "
          "conversation-level bootstrap (turns within a conversation are correlated, so a "
          "per-turn CI would be far too narrow). flip% is the fraction of items where the two "
          "presentation orders disagreed on a winner -- read it before any win rate.")

    for r in results.values():
        if (r["order_flip_disagreement"] or 0) > 0.3:
            print(f"\nWARNING: {r['arm_a']} vs {r['arm_b']} [{r['mode']}] flipped on "
                  f"{100*r['order_flip_disagreement']:.0f}% of items. That is a large position "
                  f"bias; treat the win rate as unreliable.")
        if r["arm_a"] == r["arm_b"]:
            lo, hi = r["ci95"]
            verdict = "PASS" if lo <= 0.5 <= hi else "FAIL"
            print(f"\nNEGATIVE CONTROL {r['arm_a']} vs itself [{r['mode']}]: win rate "
                  f"{r['win_rate_a']:.3f} CI [{lo:.3f},{hi:.3f}], ties "
                  f"{100*(r['tie_fraction'] or 0):.0f}% -> {verdict}")
            if verdict == "FAIL":
                print("  The judge distinguishes identical arms. No other number here can be "
                      "trusted until this is understood.")

    out = args.out or os.path.join(args.batch_dir, "results.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common_prepare(p):
        p.add_argument("--generations", required=True)
        p.add_argument("--out_dir", required=True)
        p.add_argument("--n", type=int, default=300)
        p.add_argument("--modes", default="off,on")
        p.add_argument("--batch_size", type=int, default=25)
        p.add_argument("--length_strata", default=None,
                       help="turn_token_lengths.json from the sampler.")
        p.add_argument("--defect_labels", default=None,
                       help="Output of eval/label_context_defects.py, if available.")
        p.add_argument("--strata", default="depth,length")

    p = sub.add_parser("prepare-quality"); common_prepare(p)
    p.add_argument("--pairs", required=True, help="Comma-separated arm_a:arm_b.")
    p.set_defaults(func=cmd_prepare_quality)

    p = sub.add_parser("prepare-coverage"); common_prepare(p)
    p.add_argument("--goal_contexts", required=True)
    p.add_argument("--arms", default="vanilla,prompted,distilled-fw,distilled-lora")
    p.set_defaults(func=cmd_prepare_coverage)

    p = sub.add_parser("prepare-trace"); common_prepare(p)
    p.add_argument("--arms", default="vanilla,prompted,distilled-fw,distilled-lora")
    p.set_defaults(func=cmd_prepare_trace)

    p = sub.add_parser("aggregate-quality")
    p.add_argument("--batch_dir", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--judge_batches", default="",
                   help="Explicit judge grouping as semicolon-separated batch-index lists, e.g. "
                        "'0,1,6;2,3,7;4,5,8;9,10,11'. Use this when batches were NOT handed out "
                        "contiguously -- --batches_per_judge assumes contiguity and will mislabel "
                        "which pairings were judged independently.")
    p.add_argument("--batches_per_judge", type=int, default=1,
                   help="How many consecutive batch files each judge was handed. Used to decide "
                        "whether a pairing's two orders were judged independently.")
    p.set_defaults(func=cmd_aggregate_quality)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

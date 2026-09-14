"""Render a heavy GOOD trace (trace_{conv_id}.json from precompute_good_contexts.py
--trace_dir) into a compact per-turn markdown digest.

Raw traces are large (~32 atomic + a dozen set comparisons per turn, full goal-set
distribution, generation candidates). The digest keeps exactly what the focus/
promotion investigation needs -- injected context, the FULL ranked set distribution
with Beta(alpha,beta), the top atomic goals by learned score (flagging ones new this
turn), and a one-line summary of generation/prune events -- so an analysis subagent
can read a whole conversation without drowning in raw comparisons.

Usage:
    python data/render_good_trace.py trace_<id>.json            # -> stdout
    python data/render_good_trace.py traces_gemini/ --out digests_gemini/
    python data/render_good_trace.py --pair traces_gemini/trace_X.json traces_qwen/trace_X.json
"""

import argparse
import json
import os
from collections import Counter


def _conf_str(mean, lower, upper):
    """Match format_goals_for_context's displayed convention exactly (floored
    percentages, std shown as half the ±1sigma bound width)."""
    return f"{int(mean*100)}% ± {int((upper - lower) / 2 * 100)}%"


def _goals_join(goals, limit=6):
    goals = goals or []
    shown = "; ".join(goals[:limit])
    if len(goals) > limit:
        shown += f"; (+{len(goals) - limit} more)"
    return shown


def _events_by_kind(events):
    out = {}
    for e in events:
        out.setdefault(e["kind"], []).append(e)
    return out


def render_turn(turn: dict) -> str:
    t = turn["turn_index"]
    snap = turn["snapshot"]
    ev = _events_by_kind(turn.get("events", []))
    lines = [f"### Turn {t}", ""]

    # Injected context (the actual teacher output for this turn)
    ctx = (turn.get("context") or "").strip()
    lines.append("**Injected context:**")
    lines.append("```")
    lines.append(ctx if ctx else "(empty -- no goal sets yet)")
    lines.append("```")
    lines.append("")

    # Focus + full set distribution
    sets = snap.get("sets", [])
    focus = snap.get("focus")
    if focus:
        top = sets[0]
        lines.append(f"**Focus:** {_goals_join(focus)}  "
                     f"[{_conf_str(top['mean'], top['lower'], top['upper'])}, "
                     f"α={top['alpha']} β={top['beta']}]")
    else:
        lines.append("**Focus:** (none)")
    lines.append("")
    lines.append(f"**Set distribution ({snap.get('num_sets', 0)} sets, ranked by lower bound):**")
    lines.append("")
    lines.append("| # | conf | α | β | goals |")
    lines.append("|---|------|---|---|-------|")
    for i, s in enumerate(sets):
        lines.append(f"| {i+1} | {_conf_str(s['mean'], s['lower'], s['upper'])} | {s['alpha']} "
                     f"| {s['beta']} | {_goals_join(s['goals'])} |")
    lines.append("")

    # Atomic pool by learned score, flag new-this-turn
    cur_round = snap.get("current_round")
    scores = ev.get("atomic_scores", [])
    if scores:
        ranked = scores[-1].get("ranked", [])
        lines.append(f"**Atomic pool top by score ({snap.get('num_atomic', 0)} total; "
                     f"★ = new this turn):**")
        for g in ranked[:10]:
            new = "★" if g.get("created_round") == cur_round else " "
            lines.append(f"- {new} `{g['score']:+.2f}` (lik {g['likelihood']:+.2f} / div "
                         f"{g['diversity']:+.2f}) {g['text']}")
        lines.append("")

    # Event summary line
    summary = []
    for prop in ev.get("atomic_proposals", []):
        if prop.get("count"):
            summary.append(f"proposed {prop['count']} atomic goals")
    sp = ev.get("set_proposal", [])
    if sp:
        c = Counter(p["added"] for p in sp)  # True/False/None
        summary.append(f"sets: {c.get(True,0)} added / {c.get(False,0)} dup / {c.get(None,0)} rejected")
    for r in ev.get("reindex", []):
        summary.append(f"reindexed {r['count']} set-goals→pool")
    for r in ev.get("rank_goal_sets", []):
        summary.append(f"{r['num_pairs']} set comparisons")
    for r in ev.get("atomic_comparisons", []):
        summary.append(f"{r['num_pairs']} atomic comparisons ({r['num_ties']} ties)")
    for r in ev.get("atomic_prune", []):
        summary.append(f"pruned {r['num_removed']} atomic")
    for r in ev.get("propagate_to_goal_sets", []):
        if r.get("num_removed"):
            summary.append(f"propagated {r['num_removed']} removals to sets")
    for r in ev.get("prune_goal_sets", []):
        if r.get("dropped"):
            summary.append(f"dropped {len(r['dropped'])} sets")
    if summary:
        lines.append("**Events:** " + "; ".join(summary))
        lines.append("")

    # Newly added set candidates (topic-tracking signal at the set layer)
    added = [p["goals"] for p in sp if p.get("added") and p.get("goals")]
    if added:
        lines.append("**New set candidates this turn:**")
        for g in added:
            lines.append(f"- {_goals_join(g)}")
        lines.append("")

    return "\n".join(lines)


def render_trace(trace: dict) -> str:
    head = [f"# Trace: {trace.get('conversation_id')}  (model: {trace.get('model')})", ""]
    body = [render_turn(t) for t in trace.get("turns", [])]
    return "\n".join(head) + "\n" + "\n---\n\n".join(body) + "\n"


def _load(path):
    with open(path) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="trace_*.json file or a directory of them")
    ap.add_argument("--out", default=None, help="output dir (for directory input) or file")
    args = ap.parse_args()

    if os.path.isdir(args.path):
        out_dir = args.out or (args.path.rstrip("/") + "_digests")
        os.makedirs(out_dir, exist_ok=True)
        n = 0
        for fn in sorted(os.listdir(args.path)):
            if not (fn.startswith("trace_") and fn.endswith(".json")):
                continue
            md = render_trace(_load(os.path.join(args.path, fn)))
            base = fn[len("trace_"):-len(".json")]
            with open(os.path.join(out_dir, f"digest_{base}.md"), "w") as f:
                f.write(md)
            n += 1
        print(f"Wrote {n} digests to {out_dir}")
    else:
        md = render_trace(_load(args.path))
        if args.out:
            with open(args.out, "w") as f:
                f.write(md)
            print(f"Wrote {args.out}")
        else:
            print(md)


if __name__ == "__main__":
    main()

"""The ONE place that renders "the conversation" for any judge or rater who must not see
injected goal context. Import this; do not reimplement any of it.

Why this module exists: the same contamination class -- injected goal context reaching a
judge that was supposed to be blind to it -- has now shipped three separate times, each in a
fresh ad hoc reimplementation of rendering that already existed correctly somewhere else:

  1. 2026-08-12: the quality-judge harness left the system-message goal block in the judged
     transcript for `sys-start`/`sys-late` arms (fixed by `_judged_prefix`, see the scope
     warning in EVAL_RESULTS.md).
  2. 2026-08-31: `build_mechanism_eval_items.py` extracted "the goal context" only from a
     system message, so user-fused arms showed raters an empty context block on 100% of items.
  3. 2026-09-04: `build_pairwise_items.py` rendered "the conversation" from a context-bearing
     arm's own row, leaking the notes into 100% of a "blind" item set and manufacturing a
     0.578 teacher-vs-vanilla win that vanished (0.462) on the fixed re-run. See
     SHARD0_FINDINGS.md.

The functions here are metadata-driven (each generation row records `goal_context_chars` and
`goal_context_placement`), not marker-heuristic-driven. The leak scan is the belt-and-braces
layer on top and is meant to be called on every rendered artifact before it is dispatched.
"""

from __future__ import annotations

import glob
import os

# Lead phrase of every goal-context template since v2 (render_v2.py and all variants A-F).
# Extend this list if a future template stops carrying it -- and add a test.
LEAK_MARKERS = ("Inferred notes about this user",)

# Substring probes taken from an actual goal-context string, used when the caller can supply
# the context: catches template rewrites that drop every static marker.
_PROBE_LEN = 60
_N_PROBES = 4


def judged_prefix(row: dict) -> list:
    """Everything before the final user turn, with any injected goal context removed.

    Handles the "sys-start"/"sys-late" placements, where the injected block is a separate
    SYSTEM message. Dropping every system message is safe and was verified rather than
    assumed: no eval conversation carries a system turn of its own (0/174 vanilla rows at
    each of 8B/14B/32B), so the only system message that can appear is the injected one.
    """
    return [m for m in row["prompt_messages"][:-1] if m.get("role") != "system"]


def final_user_turn(row: dict) -> dict:
    """The final user message WITHOUT any injected goal context.

    Placement-aware: the "user" placement fuses the block into the final user turn
    (`build_teacher_messages` appends "\\n\\n" + context), so exactly that many characters
    are stripped off the end; system placements leave the user turn untouched, and stripping
    there would amputate the tail of the user's real question. `goal_context_placement` is
    absent on rows generated before 2026-08-11; defaulting it to "user" reproduces the old
    behaviour exactly on those files.
    """
    msg = dict(row["prompt_messages"][-1])
    n = row.get("goal_context_chars") or 0
    placement = row.get("goal_context_placement") or "user"
    if n and placement == "user":
        content = msg.get("content") or ""
        msg["content"] = content[: max(0, len(content) - n - 2)]
    return msg


def stripped_messages(row: dict) -> list:
    """The full judged conversation for this row: prefix + cleaned final user turn."""
    return judged_prefix(row) + [final_user_turn(row)]


def render_markdown_conversation(row: dict) -> str:
    """Markdown transcript (### ROLE blocks) of the stripped conversation.

    This is the only sanctioned way to put "the full conversation" into a rater-facing
    item file. It never renders from raw `prompt_messages`.
    """
    parts = []
    for m in stripped_messages(row):
        parts.append(f"### {m['role'].upper()}\n{m['content']}")
    return "\n\n".join(parts)


def _context_probes(goal_context: str) -> list[str]:
    ctx = (goal_context or "").strip()
    if len(ctx) < _PROBE_LEN:
        return [ctx] if ctx else []
    step = max(1, (len(ctx) - _PROBE_LEN) // max(1, _N_PROBES - 1))
    return [ctx[i : i + _PROBE_LEN] for i in range(0, len(ctx) - _PROBE_LEN + 1, step)][:_N_PROBES]


def scan_text_for_leaks(text: str, goal_context: str | None = None,
                        extra_markers: tuple = ()) -> list[str]:
    """Return a list of leak reasons found in `text` (empty list = clean).

    Checks the static template markers always, and -- when the caller can supply the actual
    goal-context string for this item -- several exact substrings of it, which survives any
    future template rewording.
    """
    reasons = []
    for marker in tuple(LEAK_MARKERS) + tuple(extra_markers):
        if marker and marker in text:
            reasons.append(f"static marker present: {marker!r}")
    if goal_context:
        for probe in _context_probes(goal_context):
            if probe and probe in text:
                reasons.append(f"goal-context substring present: {probe[:40]!r}...")
                break
    return reasons


def assert_items_clean(items_dir: str, context_by_item: dict | None = None,
                       pattern: str = "item_*.md") -> int:
    """Hard-fail if any rendered item leaks. Returns the number of files scanned.

    `context_by_item` optionally maps item_id (file stem) -> that item's actual goal-context
    string for the stronger substring check. Raise, don't warn: a leaked "blind" item set is
    worse than no item set (SHARD0_FINDINGS.md §1).
    """
    dirty = {}
    files = sorted(glob.glob(os.path.join(items_dir, pattern)))
    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        text = open(path, errors="replace").read()
        ctx = (context_by_item or {}).get(stem)
        reasons = scan_text_for_leaks(text, ctx)
        if reasons:
            dirty[stem] = reasons
    if dirty:
        listing = "\n".join(f"  {k}: {'; '.join(v)}" for k, v in sorted(dirty.items())[:20])
        raise RuntimeError(
            f"LEAK SCAN FAILED: {len(dirty)}/{len(files)} rendered items in {items_dir} "
            f"contain goal-context text a blind rater must not see:\n{listing}\n"
            f"Do NOT dispatch these items. Fix the rendering (use "
            f"item_rendering.render_markdown_conversation) and rebuild.")
    return len(files)

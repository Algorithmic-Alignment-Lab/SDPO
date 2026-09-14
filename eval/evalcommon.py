"""Shared helpers for the eval harness: determinism, strata, and clustered statistics.

Three things here are easy to get wrong in ways that quietly corrupt results, so they live in
one place rather than being reimplemented per script:

1. **Determinism.** Python's builtin `hash()` on str is salted per process, so anything using it
   for sampling or seeding would silently differ between runs. Everything here goes through
   `stable_int`.
2. **Strata.** Turn depth and prompt length are reported alongside every headline number
   (prompt length especially: 26% of eval turns are longer than any prompt training saw, since
   training filtered candidate turns to <=2048 tokens). Defining the buckets once stops two
   scripts from disagreeing about what "deep" means.
3. **Clustered CIs.** Turns within a conversation are correlated. A per-turn bootstrap would
   produce intervals that are far too narrow and make noise look significant, so resampling is
   at the CONVERSATION level.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics

DEPTH_BUCKETS = ((1, 2, "d1-2"), (3, 4, "d3-4"), (5, 10**9, "d5+"))


def stable_int(*parts: str) -> int:
    """Process-independent integer from strings. Never use builtin hash() for this."""
    return int(hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:8], 16)


def stable_shuffle(items: list, *salt: str) -> list:
    """Deterministic shuffle: sort by a stable per-item hash."""
    return sorted(items, key=lambda it: stable_int(repr(it), *salt))


def load_jsonl(path: str) -> list[dict]:
    """Read newline-delimited JSON, tolerating a torn final line from a killed job."""
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def depth_bucket(turn_index: int) -> str:
    for lo, hi, name in DEPTH_BUCKETS:
        if lo <= turn_index <= hi:
            return name
    return "d?"


def load_length_strata(path: str | None) -> dict[str, dict]:
    """turn_token_lengths.json written by data/sample_wildchat_eval.py, or {} if absent."""
    if not path or not os.path.exists(path):
        return {}
    return json.load(open(path))


def length_stratum(lengths: dict, conversation_id: str, turn_index: int) -> str:
    """'fits' / 'long' relative to training's 2048-token candidate-turn filter.

    Training only ever saw prompts that fit, so `long` turns are off-distribution in length
    independently of anything about goal inference. Pooled into the headline number by decision,
    but always reported as a stratum so a length effect cannot masquerade as a quality effect.
    """
    rec = lengths.get(f"{conversation_id}:{turn_index}")
    if rec is None:
        return "unknown"
    return "fits" if rec.get("fits") else "long"


def item_strata(row: dict, lengths: dict, defects: dict | None = None) -> dict:
    """The stratification labels attached to one (conversation, turn)."""
    key = f"{row['conversation_id']}:{row['turn_index']}"
    out = {
        "depth": depth_bucket(row["turn_index"]),
        "length": length_stratum(lengths, row["conversation_id"], row["turn_index"]),
    }
    if defects is not None:
        labels = (defects.get(key) or {}).get("labels") or []
        out["defect"] = "clean" if not labels or labels == ["clean"] else "defective"
        out["defect_labels"] = sorted(labels)
    return out


def stratified_sample(items: list[dict], n: int, strata_keys: tuple[str, ...],
                      salt: str) -> list[dict]:
    """Deterministically take ~n items, spread proportionally across strata.

    Proportional rather than equal allocation: the goal is a sample that represents the eval
    set while guaranteeing thin strata are not wiped out by chance. Any stratum with at least
    one item keeps at least one.
    """
    if n >= len(items):
        return stable_shuffle(items, salt)

    groups: dict[tuple, list[dict]] = {}
    for it in items:
        groups.setdefault(tuple(str(it["strata"].get(k)) for k in strata_keys), []).append(it)

    total = len(items)
    picked: list[dict] = []
    for gk in sorted(groups):
        pool = stable_shuffle(groups[gk], salt, *gk)
        take = max(1, round(n * len(pool) / total))
        picked.extend(pool[:take])

    picked = stable_shuffle(picked, salt, "trim")
    return picked[:n]


def bootstrap_ci_by_conversation(values_by_conv: dict[str, list[float]], n_boot: int = 2000,
                                 alpha: float = 0.05, salt: str = "boot") -> tuple:
    """(point, lo, hi) for a mean, resampling CONVERSATIONS with replacement.

    Turns inside a conversation share a topic, a user, and a goal state, so they are not
    independent observations. Resampling turns directly would understate the variance; the
    cluster bootstrap resamples whole conversations and recomputes the pooled mean.

    Deterministic: the resample indices come from stable_int, so a rerun reproduces the CI.
    """
    convs = sorted(values_by_conv)
    flat = [v for c in convs for v in values_by_conv[c]]
    if not flat:
        return (float("nan"), float("nan"), float("nan"))
    point = statistics.fmean(flat)
    if len(convs) < 2:
        return (point, float("nan"), float("nan"))

    n = len(convs)
    means = []
    for b in range(n_boot):
        acc, cnt = 0.0, 0
        for j in range(n):
            c = convs[stable_int(salt, b, j) % n]
            vals = values_by_conv[c]
            acc += sum(vals)
            cnt += len(vals)
        if cnt:
            means.append(acc / cnt)
    means.sort()
    lo = means[int((alpha / 2) * len(means))]
    hi = means[min(len(means) - 1, int((1 - alpha / 2) * len(means)))]
    return (point, lo, hi)


def format_transcript(messages: list[dict], max_chars_per_msg: int = 1600) -> str:
    """Render a conversation prefix for a judge/analysis packet.

    Truncates individual messages rather than the transcript as a whole: the judge needs the
    LATEST user turn intact (it is what the response answers), and dropping from the end would
    remove exactly that.
    """
    lines = []
    turn = 0
    for m in messages:
        content = " ".join((m.get("content") or "").split())
        if len(content) > max_chars_per_msg:
            content = content[:max_chars_per_msg] + " …[truncated]"
        if m["role"] == "user":
            turn += 1
            lines.append(f"**User (turn {turn}):** {content}")
        elif m["role"] == "assistant":
            lines.append(f"**Assistant:** {content}")
        else:
            lines.append(f"**{m['role'].title()}:** {content}")
    return "\n\n".join(lines)

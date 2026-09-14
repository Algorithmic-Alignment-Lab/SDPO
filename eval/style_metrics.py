"""Measurement axis 4: cheap automatic style/form metrics over generated responses.

These exist to make one specific failure mode detectable. Context distillation is prone to
transferring the teacher's *form* without its *function* -- the student learns to sound like a
goal-aware model (longer, more hedged, more list-structured, more clarifying questions) without
actually inferring goals any better than the base model. A naive pairwise win rate cannot tell
that apart from real improvement, because both produce the same "distilled beats vanilla" number.

So: quality and goal-coverage are judged (elsewhere), and *form* is measured here, cheaply and
deterministically, with no judge in the loop. A `distilled > vanilla` result whose entire
signature is longer and more hedged prose gets reported as a style artifact.

**Goal-text leakage: a cheap null-guard, NOT an expected finding.** Read this before drawing
anything from those columns.

MEASURED 2026-08-05 over 5,865 generations, and the original prediction here was WRONG. I argued
that because the injected context (`format_goals_for_context`) is purely declarative -- a
"## User Goals (for context)" header over "**Plausible concerns** (avoid violating)" bullets, with
no instruction to restate them -- leakage would read ~0 for every arm. It does not.

Length-controlled results (mean fraction of a turn's goal bullets reproduced, mode=off, matched
word-count bands, degenerate replies excluded):

    band          vanilla  placebo  prompted  distilled-lora  distilled-fw
    600-1000w      0.095    0.082     0.288        0.136          0.181
    1000-1600w     0.179    0.082     0.346        0.160          0.195

Read it in this order:
  * Length matters a lot on its own -- vanilla climbs 0.078 -> 0.095 -> 0.179 across bands -- so
    NEVER compare this metric across arms without matching on length. The distilled arms are
    1.8-2.7x longer than vanilla, which alone inflates the raw numbers.
  * `prompted` is elevated at EVERY matched length (~2-3x vanilla). The teacher really does recite
    goal content. That is a property of the injection format contaminating the distillation
    target, not a student defect.
  * `prompted-placebo` sits at or BELOW vanilla (0.082) despite carrying goal-shaped text in its
    prompt. That is the control that makes the `prompted` number meaningful: the effect is about
    matching CONTENT, not about having some goal-ish block present.
  * The distilled arms look mildly above vanilla at matched length, but vanilla's own length trend
    and the thin n in its long band (n=28) mean length is NOT yet separated from a real effect.
    Do not claim internalized goal content from this without a proper length-controlled model.

Note especially that leakage is NOT expected in the teacher (`prompted`) either, so its value is
not a "ceiling" to normalise the student against. Interpret it this way:
  * `prompted` ~ 0  -- the expected case. No source for echoing exists, here or downstream.
  * `prompted` > 0  -- a finding about the INJECTION FORMAT contaminating the distillation
    target, i.e. about the teacher, not the student. Worth knowing before trusting the target.
  * `distilled-*` above `vanilla`'s coincidental base rate while `prompted` ~ 0 -- would be
    genuinely surprising, since the student is pulled toward the teacher's response
    distribution and would have no source to copy from. No mechanism predicts it.

The one concrete use even at ~0: the quality judge sees the conversation and the response but
NOT the goal context, so a response reciting goals the judge cannot see the source of would read
as unexplained padding -- biasing the quality comparison against whichever arm does it, for a
format artifact rather than a real quality difference. This detects that.

The lexical proxies (hedging, clarifying questions) are crude by construction and labelled as
such; they are screening signals for the qualitative pass, not findings on their own.

Usage:
    python eval/style_metrics.py \
        --generations eval_runs/generations.jsonl \
        --goal_contexts datasets/wildchat_eval_250/goal_contexts_235b.json \
        --out eval_runs/style_metrics.jsonl --summary
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter

# Crude lexical proxies. Kept explicit and inspectable rather than hidden in a model, but they
# are proxies: "may" also appears in ordinary prose. Report alongside, never instead of, judged
# measures.
HEDGE_PATTERNS = [
    r"\bit depends\b", r"\bmight\b", r"\bmay\b", r"\bcould\b", r"\bperhaps\b",
    r"\bpossibly\b", r"\bgenerally\b", r"\btypically\b", r"\busually\b",
    r"\bkeep in mind\b", r"\bbear in mind\b", r"\bthat said\b", r"\bhowever\b",
    r"\bon the other hand\b", r"\bi'?m not (?:entirely )?sure\b", r"\bif i understand\b",
    r"\bwithout more (?:information|context|details)\b",
]
# Second-person questions -- an attempt to distinguish "asking the user something" from a
# rhetorical or restated question.
CLARIFY_PATTERNS = [
    r"\bcould you (?:please )?(?:clarify|specify|tell me|confirm|share)\b",
    r"\bcan you (?:clarify|specify|tell me|confirm|share)\b",
    r"\bwhat (?:exactly )?(?:do|did) you (?:mean|want|have in mind)\b",
    r"\bwhich (?:one|option|approach) (?:do|would) you\b",
    r"\bare you looking for\b", r"\bdo you (?:want|need|prefer|have)\b",
    r"\bjust to (?:be sure|confirm|check)\b", r"\bto clarify\b",
]

_WORD = re.compile(r"[a-z0-9']+")
_BULLET = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+", re.MULTILINE)
_HEADER = re.compile(r"^\s*#{1,6}\s+\S", re.MULTILINE)

# Minimum words before a duplication ratio means anything; short replies trivially have few
# n-grams and would otherwise produce noisy extremes.
_REP_MIN_WORDS = 80


def repetition(text: str, n: int = 8) -> float:
    """Fraction of duplicated word n-grams: 0.0 = all distinct, ->1.0 = collapsed.

    ADDED AFTER THIS WAS CAUGHT IN REAL DATA, and it is the metric this module most needed.
    The original style axis measured length, hedging, structure, clarification and goal-text
    leakage -- but not degeneration, which turned out to be the actual difference between arms:
    on mode=off, responses with >0.60 duplicated 8-grams occurred in 6.2% of `distilled-fw` and
    8.3% of `distilled+context` outputs versus 1.0% of `vanilla`. One concrete case: asked to
    "rewrite that as if you were an Indian learning English", distilled-fw emitted a 48-item
    enumerated listicle about semaphores to the token ceiling and never addressed the request.

    Why it matters for the eval rather than just being a curiosity:
      * It is a heavy TAIL, not typical behaviour (medians are ~0.03 vs ~0.00), so median-based
        style comparisons hide it entirely -- always report the tail fractions.
      * It makes truncation rate arm-dependent, which would otherwise look like a max_tokens
        problem. Raising the cap buys longer degenerate text, not better data.
      * A degenerate response is not merely "verbose"; a quality judge will rightly punish it.
        So a `distilled` loss on quality must be checked against this before being read as
        "distillation didn't help" -- the mechanism would be generation collapse, not goal
        insensitivity.
    """
    words = _WORD.findall((text or "").lower())
    if len(words) < _REP_MIN_WORDS:
        return 0.0
    grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def norm_tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def ngrams(tokens: list[str], n: int) -> set[tuple]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def longest_shared_run(a: list[str], b: list[str], cap: int = 4000) -> int:
    """Longest run of consecutive tokens appearing in both.

    Standard DP but bounded: responses and contexts are both a few thousand tokens, and an
    unbounded O(len(a)*len(b)) table over the long tail (8k-char contexts) would dominate the
    whole pass. Truncating is safe here because we only care whether a LONG verbatim span
    exists, and 4000 tokens is far beyond any plausible quotation.
    """
    a, b = a[:cap], b[:cap]
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    best = 0
    for i in range(1, len(a) + 1):
        cur = [0] * (len(b) + 1)
        ai = a[i - 1]
        for j in range(1, len(b) + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


def context_bullets(context: str) -> list[list[str]]:
    """Goal lines from a GOOD context, as token lists.

    format_goals_for_context emits markdown bullets under headers; the bullets are the actual
    goal statements, so they are the unit a leaking response would reproduce.
    """
    out = []
    for line in (context or "").splitlines():
        if _BULLET.match(line):
            toks = norm_tokens(_BULLET.sub("", line, count=1))
            if len(toks) >= 4:  # ignore stubs, which would match trivially
                out.append(toks)
    return out


def leakage(answer: str, context: str) -> dict:
    """How much of the goal context appears in the response."""
    a_toks, c_toks = norm_tokens(answer), norm_tokens(context)
    if not a_toks or not c_toks:
        return {"leak_ngram8_frac": 0.0, "leak_max_run": 0, "leak_bullet_hits": 0,
                "leak_bullet_frac": 0.0}

    a8, c8 = ngrams(a_toks, 8), ngrams(c_toks, 8)
    ngram_frac = len(a8 & c8) / len(a8) if a8 else 0.0

    a_set = set(a_toks)
    bullets = context_bullets(context)
    # A bullet counts as reproduced when >=80% of its tokens appear in the response. Token-set
    # containment rather than exact match, so light rewording still registers.
    hits = sum(1 for b in bullets if sum(t in a_set for t in b) / len(b) >= 0.8)

    return {
        "leak_ngram8_frac": round(ngram_frac, 4),
        "leak_max_run": longest_shared_run(a_toks, c_toks),
        "leak_bullet_hits": hits,
        "leak_bullet_frac": round(hits / len(bullets), 4) if bullets else 0.0,
    }


def style(answer: str) -> dict:
    text = answer or ""
    toks = norm_tokens(text)
    n_words = max(len(toks), 1)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]

    hedges = sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in HEDGE_PATTERNS)
    clarifies = sum(len(re.findall(p, text, flags=re.IGNORECASE)) for p in CLARIFY_PATTERNS)

    rep = repetition(text)
    return {
        "chars": len(text),
        "words": len(toks),
        "sentences": len(sentences),
        "rep8": round(rep, 4),
        # Thresholded flags, because the signal lives in the tail, not the average.
        "rep8_over_15": rep > 0.15,
        "rep8_over_30": rep > 0.30,
        "rep8_degenerate": rep > 0.60,
        "questions": text.count("?"),
        "asks_clarifying_question": bool(clarifies) or text.rstrip().endswith("?"),
        "clarify_hits": clarifies,
        # Per-1000-words so length does not drive the rate. Length is reported separately.
        "hedge_per_1k_words": round(1000 * hedges / n_words, 2),
        "hedge_hits": hedges,
        "bullets": len(_BULLET.findall(text)),
        "headers": len(_HEADER.findall(text)),
        "has_structure": bool(_BULLET.search(text) or _HEADER.search(text)),
    }


def summarize(rows: list[dict]) -> None:
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["arm"], r["mode"]), []).append(r)

    def med(rs, k):
        vals = [r[k] for r in rs if r.get(k) is not None]
        return statistics.median(vals) if vals else 0

    print(f"\n{'arm':<19} {'mode':<5} {'n':>5} {'words':>7} {'hedge/1k':>9} {'clarify%':>9} "
          f"{'struct%':>8} {'leak8%':>7} {'leakBul%':>9} {'rep>.15':>8} {'rep>.30':>8} {'DEGEN%':>7}")
    for gk in sorted(groups):
        rs = groups[gk]
        print(f"{gk[0]:<19} {gk[1]:<5} {len(rs):>5} {med(rs,'words'):>7.0f} "
              f"{med(rs,'hedge_per_1k_words'):>9.1f} "
              f"{100*sum(r['asks_clarifying_question'] for r in rs)/len(rs):>9.1f} "
              f"{100*sum(r['has_structure'] for r in rs)/len(rs):>8.1f} "
              f"{100*med(rs,'leak_ngram8_frac'):>7.2f} "
              f"{100*med(rs,'leak_bullet_frac'):>9.2f} "
              f"{100*sum(r['rep8_over_15'] for r in rs)/len(rs):>8.1f} "
              f"{100*sum(r['rep8_over_30'] for r in rs)/len(rs):>8.1f} "
              f"{100*sum(r['rep8_degenerate'] for r in rs)/len(rs):>7.1f}")

    # Degeneration gets its own explicit comparison against vanilla, because it is the one style
    # measure that can invalidate a quality result outright rather than merely qualifying it.
    for mode in sorted({gk[1] for gk in groups}):
        van = groups.get(("vanilla", mode))
        if not van:
            continue
        base = 100 * sum(r["rep8_degenerate"] for r in van) / len(van)
        for gk in sorted(groups):
            if gk[1] != mode or gk[0] == "vanilla":
                continue
            rs = groups[gk]
            rate = 100 * sum(r["rep8_degenerate"] for r in rs) / len(rs)
            if rate > max(2.0, 2 * base):
                print(f"\nDEGENERATION [{gk[0]}/{mode}]: {rate:.1f}% of responses have >60% "
                      f"duplicated 8-grams vs {base:.1f}% for vanilla. These are collapsed "
                      f"generations, not verbose ones. Do NOT read a quality loss for this arm as "
                      f"goal-insensitivity until they are excluded or reported separately, and do "
                      f"NOT raise max_tokens -- that buys longer degenerate text.")

    leak_max = max((med(rs, "leak_bullet_frac") for rs in groups.values()), default=0)
    if leak_max <= 0.02:
        print("\nLeakage: ~0 across all arms, which is the expected result -- the injected goal "
              "context is declarative and never asks the model to restate it. Nothing to see "
              "here; this column is a null-guard, not a finding.")
    else:
        print("\nLeakage is NON-ZERO, which was not expected. Check `prompted` FIRST: elevated "
              "leakage there is a finding about the injection format contaminating the "
              "distillation target (a teacher problem, not a student one), and it also biases "
              "the quality judge, which never sees the goal context and so reads recited goals "
              "as unexplained padding. Only if `prompted` is ~0 while `distilled-*` exceeds "
              "`vanilla`'s coincidental base rate is this about the student -- and no mechanism "
              "predicts that.")

    # Flag the specific pattern that would mean "style transfer, not goal-sensitivity".
    for mode in sorted({gk[1] for gk in groups}):
        van = groups.get(("vanilla", mode))
        for arm in ("distilled-fw", "distilled-lora"):
            dis = groups.get((arm, mode))
            if not van or not dis:
                continue
            dv, vv = med(dis, "words"), med(van, "words")
            if vv and dv / vv > 1.25:
                print(f"\nNOTE [{arm}/{mode}]: median length is {dv/vv:.2f}x vanilla. Report "
                      f"length-controlled win rates for this arm; a length-driven judge "
                      f"preference would otherwise read as a quality gain.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--goal_contexts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    contexts = json.load(open(args.goal_contexts))
    rows = []
    n_err = 0
    with open(args.generations) as fh, open(args.out, "w") as out:
        for line in fh:
            try:
                gen = json.loads(line)
            except Exception:  # noqa: BLE001 - torn final line from a killed job
                continue
            if gen.get("error"):
                n_err += 1
                continue
            ctx = contexts.get(f"{gen['conversation_id']}:{gen['turn_index']}", "")
            row = {
                "conversation_id": gen["conversation_id"],
                "turn_index": gen["turn_index"],
                "arm": gen["arm"],
                "mode": gen["mode"],
                "sample": gen["sample"],
                "think_chars": len(gen.get("think") or ""),
                **style(gen.get("answer", "")),
                **leakage(gen.get("answer", ""), ctx),
            }
            rows.append(row)
            out.write(json.dumps(row) + "\n")

    print(f"wrote {args.out}: {len(rows)} rows ({n_err} generation errors skipped)")
    print(f"arms x modes: {sorted(Counter((r['arm'], r['mode']) for r in rows).items())[:4]} ...")
    if args.summary:
        summarize(rows)


if __name__ == "__main__":
    main()

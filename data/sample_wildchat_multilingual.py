"""Sample a small, language-stratified slice of WildChat-1M for the GOOD
goal-tracking diagnostic.

Unlike preprocess_wildchat_good_smoke.py (which hard-filters to English), this
stratifies across a target language list so the diagnostic exercises GOOD's
partial-localization failure mode alongside the topic-tracking one. Biases toward
longer conversations (--min_turns) so topics actually evolve / detour within a
conversation.

Writes, into --output_dir:
  - conversations.json : {conversation_id: [{turn_index, messages}]} -- the exact
    schema precompute_good_contexts.py consumes (messages = prefix through user_k).
  - languages.json     : {conversation_id: language} -- so the analysis knows what
    language each conversation's goals *should* be written in.
"""

import argparse
import json
import math
import os
from collections import defaultdict

import datasets

# A diverse default spread: Latin + non-Latin scripts, LTR + RTL. WildChat's
# `language` field is a language name string (e.g. "English", "Chinese").
DEFAULT_LANGUAGES = [
    "English", "Chinese", "Russian", "Japanese", "Spanish",
    "Portuguese", "French", "German", "Korean", "Arabic", "Turkish", "Italian",
]


def explode_conversation(conversation: list[dict]) -> list[dict]:
    """One entry per turn boundary k: messages = everything through user_k.

    Mirrors preprocess_wildchat_good_smoke.explode_conversation's prompt slicing
    (conversation[:2k-1]) so the GOOD walk sees the same prefixes training does.
    """
    num_turns = len(conversation) // 2
    out = []
    for k in range(1, num_turns + 1):
        prefix = conversation[: 2 * k - 1]
        out.append({
            "turn_index": k,
            "messages": [{"role": m["role"], "content": m["content"]} for m in prefix],
        })
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="datasets/wildchat_good_diag30")
    parser.add_argument("--num_conversations", type=int, default=30)
    parser.add_argument("--min_turns", type=int, default=5)
    parser.add_argument("--languages", default=",".join(DEFAULT_LANGUAGES),
                        help="Comma-separated WildChat language names to stratify across.")
    parser.add_argument("--max_scan", type=int, default=200000,
                        help="Safety cap on rows streamed while filling buckets.")
    parser.add_argument("--seed_skip", type=int, default=0,
                        help="Skip this many qualifying rows before collecting (vary the sample).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    languages = [s.strip() for s in args.languages.split(",") if s.strip()]
    per_lang = math.ceil(args.num_conversations / len(languages))

    print(f"Streaming allenai/WildChat-1M; stratifying across {len(languages)} languages "
          f"(<= {per_lang}/language, {args.num_conversations} total, min_turns={args.min_turns})")
    ds = datasets.load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    buckets: dict[str, list] = defaultdict(list)
    total = 0
    scanned = 0
    skipped = 0
    for row in ds:
        scanned += 1
        if scanned > args.max_scan:
            print(f"Hit --max_scan={args.max_scan}; stopping early.")
            break
        lang = row.get("language")
        if lang not in languages:
            continue
        if row["turn"] < args.min_turns:
            continue
        if any(m["toxic"] or m["redacted"] for m in row["conversation"]):
            continue
        if len(buckets[lang]) >= per_lang:
            continue
        if skipped < args.seed_skip:
            skipped += 1
            continue
        buckets[lang].append(row)
        total += 1
        # The per-language cap forces diversity: common languages (English) top out
        # at per_lang, so reaching the total requires pulling from many languages.
        if total >= args.num_conversations:
            break

    collected = [r for lang in languages for r in buckets[lang]]
    collected = collected[: args.num_conversations]
    got = {lang: len(buckets[lang]) for lang in languages if buckets[lang]}
    print(f"Collected {len(collected)} conversations after scanning {scanned} rows. "
          f"Per language: {got}")

    conversations_lookup = {}
    languages_lookup = {}
    for row in collected:
        conv_hash = row["conversation_hash"]
        conversations_lookup[conv_hash] = explode_conversation(row["conversation"])
        languages_lookup[conv_hash] = row.get("language")

    with open(os.path.join(args.output_dir, "conversations.json"), "w") as f:
        json.dump(conversations_lookup, f)
    with open(os.path.join(args.output_dir, "languages.json"), "w") as f:
        json.dump(languages_lookup, f, ensure_ascii=False, indent=2)

    print(f"Wrote {args.output_dir}/conversations.json ({len(conversations_lookup)} conversations) "
          f"and languages.json")


if __name__ == "__main__":
    main()

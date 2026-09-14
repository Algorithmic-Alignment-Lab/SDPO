"""Draw a fresh, provably-disjoint WildChat evaluation set for the GOOD-distillation eval.

Why a fresh draw rather than the existing 50-conversation val split. The split itself is
clean (whole-conversation, deterministic, `train ∩ val = ∅`), but two things argue against
reusing it as the primary eval set:

  1. **Size.** 50 conversations / 283 turns is thin for a powered pairwise rubric, once it
     is sliced by turn depth and context-defect label.
  2. **It is not hermetically sealed.** 1 of the 50 val conversations also appears in
     `datasets/wildchat_good_diag30`, and all 720 `datasets/gepa_judge/instances.json`
     comparison instances were mined from those 30 diagnostic conversations. So that
     conversation informed GOOD scaffold tuning and judge calibration -- a small leak, but
     a real one, and free to avoid.

Distribution matching is by construction, not by hope: this applies the *same* filters the
training pool used (`data/preprocess_wildchat_good_smoke.py` -- English, `turn >= 3`, no
toxic/redacted message) over the same stream, and simply skips every conversation hash
already spoken for. Because the training pool took the *first* N matching rows, excluding
its hashes naturally continues from where it stopped -- but this excludes by hash rather
than by row offset, so it stays correct even if the stream order ever changes.

Output is `conversations.json` in the `{conversation_id: [{turn_index, messages}]}` schema
that `data/precompute_good_contexts.py` and the eval generation driver both consume, plus a
`manifest.json` recording provenance and the distributions the analysis stratifies on.

Usage:
    python data/sample_wildchat_eval.py \
        --output_dir datasets/wildchat_eval_250 \
        --num_conversations 250 \
        --exclude datasets/wildchat_good_1k/conversations.json \
        --exclude datasets/wildchat_good_1k/conversations_train.json \
        --exclude datasets/wildchat_good_1k/conversations_val.json \
        --exclude datasets/wildchat_good_diag30/conversations.json \
        --exclude datasets/gepa_judge/instances.json \
        --tokenizer Qwen/Qwen3-8B --max_prompt_length 2048
"""

import argparse
import json
import os
import statistics
import sys

import datasets

from preprocess_wildchat_good_smoke import explode_conversation


def load_conversation_ids(path: str) -> set[str]:
    """Collect conversation ids from any of the shapes our artifacts use.

    Handles both the `{conversation_id: ...}` conversation lookups and the
    `[{"conversation_id": ..., ...}, ...]` mined-instance lists (gepa_judge), so a caller
    can pass every artifact that must be excluded without knowing its layout.
    """
    with open(path) as f:
        blob = json.load(f)

    if isinstance(blob, dict):
        return set(blob)
    if isinstance(blob, list):
        ids = {
            item["conversation_id"]
            for item in blob
            if isinstance(item, dict) and item.get("conversation_id")
        }
        if not ids:
            raise ValueError(f"{path}: list contained no conversation_id fields")
        return ids
    raise ValueError(f"{path}: unsupported JSON layout {type(blob).__name__}")


def token_lengths(conversations: dict, tokenizer, max_prompt_length: int) -> dict:
    """Per-turn prompt token lengths, for the context-length stratification.

    Deliberately does NOT drop long turns. Training filtered candidate turns to those
    fitting `max_prompt_length` (WildChatChopDataset), so whether the eval should match
    that filter is an analysis decision, not a sampling one -- recording a `fits` flag per
    turn keeps both options open instead of silently discarding data here.
    """
    lengths, n_fit, n_total = [], 0, 0
    fits_by_turn = {}
    for conv_id, turns in conversations.items():
        for t in turns:
            n = len(tokenizer.apply_chat_template(t["messages"], add_generation_prompt=True))
            fits = n <= max_prompt_length
            fits_by_turn[f"{conv_id}:{t['turn_index']}"] = {"prompt_tokens": n, "fits": fits}
            lengths.append(n)
            n_total += 1
            n_fit += int(fits)

    lengths.sort()
    return {
        "per_turn": fits_by_turn,
        "summary": {
            "turns": n_total,
            "fits_max_prompt_length": n_fit,
            "max_prompt_length": max_prompt_length,
            "min": lengths[0],
            "p50": lengths[len(lengths) // 2],
            "p90": lengths[int(len(lengths) * 0.9)],
            "max": lengths[-1],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default="datasets/wildchat_eval_250")
    ap.add_argument("--num_conversations", type=int, default=250)
    ap.add_argument("--min_turns", type=int, default=3, help="Match the training pool's filter.")
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="JSON artifact whose conversation ids must be excluded AND then asserted absent. "
        "Repeatable. Pass every already-used artifact: the pool, train, val, diag30, gepa_judge.",
    )
    ap.add_argument("--data_source", default="wildchat_eval")
    ap.add_argument("--tokenizer", default=None, help="Optional HF id/path for token-length stats.")
    ap.add_argument("--max_prompt_length", type=int, default=2048)
    ap.add_argument("--max_scan", type=int, default=2_000_000, help="Stream safety bound.")
    args = ap.parse_args()

    if not args.exclude:
        sys.exit(
            "refusing to draw an eval set with no --exclude artifacts: the whole point is "
            "provable disjointness from anything already trained on or tuned against"
        )

    exclude_sources = {}
    exclude_ids: set[str] = set()
    for path in args.exclude:
        ids = load_conversation_ids(path)
        exclude_sources[path] = len(ids)
        exclude_ids |= ids
        print(f"exclude: {len(ids):6d} ids from {path}")
    print(f"exclude: {len(exclude_ids)} distinct ids total")

    print(f"streaming allenai/WildChat-1M for {args.num_conversations} conversations ...")
    ds = datasets.load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    selected, scanned = [], 0
    n_skip_excluded = n_skip_turns = n_skip_lang = n_skip_toxic = n_skip_dup = 0
    seen_hashes: set[str] = set()

    for row in ds:
        scanned += 1
        if scanned > args.max_scan:
            break

        # Same filter order and semantics as preprocess_wildchat_good_smoke.py, so the eval
        # distribution matches the training pool's.
        if row["turn"] < args.min_turns:
            n_skip_turns += 1
            continue
        if row["language"] != "English":
            n_skip_lang += 1
            continue
        if any(m["toxic"] or m["redacted"] for m in row["conversation"]):
            n_skip_toxic += 1
            continue

        conv_hash = row["conversation_hash"]
        if conv_hash in exclude_ids:
            n_skip_excluded += 1
            continue
        if conv_hash in seen_hashes:
            n_skip_dup += 1
            continue

        seen_hashes.add(conv_hash)
        selected.append(row)
        if len(selected) % 25 == 0:
            print(f"  selected {len(selected)}/{args.num_conversations} (scanned {scanned})")
        if len(selected) >= args.num_conversations:
            break

    print(
        f"scanned {scanned} rows -> selected {len(selected)}; skipped: "
        f"{n_skip_turns} short, {n_skip_lang} non-English, {n_skip_toxic} toxic/redacted, "
        f"{n_skip_excluded} already-used, {n_skip_dup} duplicate-hash"
    )
    if len(selected) < args.num_conversations:
        sys.exit(f"only found {len(selected)} of {args.num_conversations}; raise --max_scan")

    conversations = {}
    for row in selected:
        conv_hash = row["conversation_hash"]
        exploded = explode_conversation(conv_hash, row["conversation"])
        conversations[conv_hash] = [
            {"turn_index": ex["extra_info"]["turn_index"], "messages": ex["prompt"]}
            for ex in exploded
        ]

    # --- BLOCKING verification. A leak here invalidates every number downstream, so this
    # runs before anything is written and exits non-zero rather than warning.
    drawn = set(conversations)
    for path in args.exclude:
        overlap = drawn & load_conversation_ids(path)
        if overlap:
            sys.exit(
                f"DISJOINTNESS VIOLATION: {len(overlap)} drawn conversation(s) also appear in "
                f"{path}: {sorted(overlap)[:5]}. Nothing written."
            )
    print(f"disjointness OK: 0 overlap with any of {len(args.exclude)} excluded artifacts")

    turn_counts = sorted(len(v) for v in conversations.values())
    manifest = {
        "data_source": args.data_source,
        "source_dataset": "allenai/WildChat-1M",
        "filters": {
            "language": "English",
            "min_turns": args.min_turns,
            "exclude_toxic_or_redacted": True,
        },
        "note": (
            "Filters mirror data/preprocess_wildchat_good_smoke.py so this set matches the "
            "training pool's distribution. Excluded by conversation_hash, not row offset."
        ),
        "exclude_sources": exclude_sources,
        "exclude_ids_total": len(exclude_ids),
        "rows_scanned": scanned,
        "skipped": {
            "short": n_skip_turns,
            "non_english": n_skip_lang,
            "toxic_or_redacted": n_skip_toxic,
            "already_used": n_skip_excluded,
            "duplicate_hash": n_skip_dup,
        },
        "conversations": len(conversations),
        "turns": sum(turn_counts),
        "turns_per_conversation": {
            "min": turn_counts[0],
            "p50": int(statistics.median(turn_counts)),
            "max": turn_counts[-1],
        },
    }

    if args.tokenizer:
        from transformers import AutoTokenizer

        print(f"computing token lengths with {args.tokenizer} ...")
        tok = AutoTokenizer.from_pretrained(args.tokenizer)
        stats = token_lengths(conversations, tok, args.max_prompt_length)
        manifest["tokenizer"] = args.tokenizer
        manifest["prompt_tokens"] = stats["summary"]
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "turn_token_lengths.json"), "w") as f:
            json.dump(stats["per_turn"], f)
        print(f"  {stats['summary']}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "conversations.json"), "w") as f:
        json.dump(conversations, f)
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"wrote {args.output_dir}/conversations.json "
        f"({len(conversations)} conversations, {sum(turn_counts)} turns) and manifest.json"
    )


if __name__ == "__main__":
    main()

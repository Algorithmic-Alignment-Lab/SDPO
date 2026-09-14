"""Small-scale WildChat-1M slice for validating the GOOD-teacher integration.

Pulls a handful of real multi-turn conversations from WildChat-1M (ungated,
ODC-BY licensed), explodes each at every turn boundary into a separate
training example (prefix = fixed prompt, next assistant turn = generated
on-policy), and writes both the exploded parquet files SDPO's RLHFDataset
expects and a conversation lookup JSON the GOOD state cache actor uses to
walk forward through a conversation's turns on a cache miss.

This is NOT the full WildChat-1M pipeline (that needs a different, scalable
lookup strategy for the cache and the row-grouped large-schema parquet
writer in data/preprocess.py) -- it's a small, real-data prototype sized for
smoke-testing the GOOD-teacher wiring itself.
"""

import argparse
import json
import os

import datasets


def explode_conversation(conversation_hash: str, conversation: list[dict]) -> list[dict]:
    """Explode one WildChat conversation into one example per turn boundary.

    `conversation` alternates user/assistant messages (user_1, assistant_1,
    user_2, assistant_2, ...). For cut point k (1-indexed), the prompt is
    everything up through user_k; the model generates assistant_k on-policy.
    """
    num_turns = len(conversation) // 2
    examples = []
    for k in range(1, num_turns + 1):
        prefix = conversation[: 2 * k - 1]
        prompt = [{"role": m["role"], "content": m["content"]} for m in prefix]
        examples.append(
            {
                "data_source": "wildchat_good_smoke",
                "prompt": prompt,
                "ability": "dialogue",
                "reward_model": {"style": "none", "ground_truth": ""},
                "extra_info": {
                    "conversation_id": conversation_hash,
                    "turn_index": k,
                },
            }
        )
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="datasets/wildchat_good_smoke")
    parser.add_argument("--num_conversations", type=int, default=30)
    parser.add_argument("--num_test_conversations", type=int, default=4)
    parser.add_argument("--min_turns", type=int, default=3)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Streaming allenai/WildChat-1M ...")
    ds = datasets.load_dataset("allenai/WildChat-1M", split="train", streaming=True)

    wanted = args.num_conversations + args.num_test_conversations
    conversations = []
    for row in ds:
        if row["turn"] < args.min_turns:
            continue
        if row["language"] != "English":
            continue
        if any(m["toxic"] or m["redacted"] for m in row["conversation"]):
            continue
        conversations.append(row)
        if len(conversations) >= wanted:
            break

    print(f"Collected {len(conversations)} conversations (wanted {wanted})")

    test_conversations = conversations[: args.num_test_conversations]
    train_conversations = conversations[args.num_test_conversations :]

    conversations_lookup = {}
    train_examples = []
    test_examples = []
    for split_conversations, examples in (
        (train_conversations, train_examples),
        (test_conversations, test_examples),
    ):
        for row in split_conversations:
            conv_hash = row["conversation_hash"]
            exploded = explode_conversation(conv_hash, row["conversation"])
            examples.extend(exploded)
            conversations_lookup[conv_hash] = [
                {"turn_index": ex["extra_info"]["turn_index"], "messages": ex["prompt"]}
                for ex in exploded
            ]

    print(f"Train: {len(train_examples)} exploded examples from {len(train_conversations)} conversations")
    print(f"Test: {len(test_examples)} exploded examples from {len(test_conversations)} conversations")

    datasets.Dataset.from_list(train_examples).to_parquet(os.path.join(args.output_dir, "train.parquet"))
    datasets.Dataset.from_list(test_examples).to_parquet(os.path.join(args.output_dir, "test.parquet"))

    with open(os.path.join(args.output_dir, "conversations.json"), "w") as f:
        json.dump(conversations_lookup, f)

    print(f"Wrote {args.output_dir}/{{train,test}}.parquet and conversations.json")


if __name__ == "__main__":
    main()

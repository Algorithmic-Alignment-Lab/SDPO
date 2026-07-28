"""Split a WildChat conversations.json into disjoint train / validation files.

Every training run so far pointed `data.train_files` and `data.val_files` at the
*same* conversations.json, so there was no held-out data at all. Since the
distilled student will eventually be compared against the live-GOOD teacher on
conversations it never trained on, that split has to exist before training --
carving it out afterwards would mean throwing away the run.

The split is on whole conversations, not turns: WildChatChopDataset keeps one
row per conversation and redraws the chop point each epoch, so splitting at the
turn level would leak earlier turns of a held-out conversation into training.

Both output files keep the input's exact
`{conversation_id: [{"turn_index": k, "messages": [...]}]}` schema, and both are
served by the same goal_contexts JSON (its keys are "{conv_id}:{turn_index}",
so nothing about the context lookup needs to change).

The split is deterministic: conversation ids are sorted before shuffling with a
seeded RNG, so re-running reproduces the same partition regardless of dict
ordering.
"""

import argparse
import json
import os
import random

ap = argparse.ArgumentParser()
ap.add_argument("--conversations_path", default="datasets/wildchat_good_1k/conversations.json")
ap.add_argument("--train_path", default=None, help="Defaults to <input dir>/conversations_train.json")
ap.add_argument("--val_path", default=None, help="Defaults to <input dir>/conversations_val.json")
ap.add_argument("--num_val", type=int, default=50, help="Number of conversations held out.")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--write", action="store_true", help="Actually write the files.")
args = ap.parse_args()

with open(args.conversations_path) as f:
    conversations = json.load(f)

ds_dir = os.path.dirname(args.conversations_path)
train_path = args.train_path or os.path.join(ds_dir, "conversations_train.json")
val_path = args.val_path or os.path.join(ds_dir, "conversations_val.json")

if args.num_val >= len(conversations):
    raise SystemExit(f"--num_val {args.num_val} must be smaller than the {len(conversations)} conversations available")

# sorted() first so the partition depends only on the ids and the seed, never on
# the order json.load happened to produce.
ids = sorted(conversations)
random.Random(args.seed).shuffle(ids)
val_ids = set(ids[: args.num_val])

train = {cid: turns for cid, turns in conversations.items() if cid not in val_ids}
val = {cid: turns for cid, turns in conversations.items() if cid in val_ids}

assert not (set(train) & set(val)), "train/val overlap"
assert len(train) + len(val) == len(conversations), "conversations lost in the split"


def _turns(d):
    return sum(len(v) for v in d.values())


print(f"input : {len(conversations)} conversations, {_turns(conversations)} turns")
print(f"train : {len(train)} conversations, {_turns(train)} turns -> {train_path}")
print(f"val   : {len(val)} conversations, {_turns(val)} turns -> {val_path}")

if args.write:
    for path, data in ((train_path, train), (val_path, val)):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
    print("\nwrote both files")
else:
    print("\n(dry run; pass --write to produce the files)")

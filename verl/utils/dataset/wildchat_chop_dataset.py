"""Dataset that resamples a fresh conversation "chop point" on every access.

The GOOD-teacher precompute produces a goal context for *every* turn of every
conversation. To make full use of that -- and to avoid letting long
conversations dominate the training mix (a 16-turn conversation would
otherwise contribute 16x the exploded examples of a 3-turn one) -- this
dataset keeps exactly **one logical row per conversation** and, on each
`__getitem__`, samples which turn boundary to cut the multi-turn conversation
at. Because a shuffled dataloader re-draws every row each epoch, this yields a
different chop for a given conversation across epochs while every draw still
lands on a fully-precomputed per-turn goal context (see
`verl/utils/good_state_cache.py`).

Source of truth is the `conversations.json` written by
`data/preprocess_wildchat_good_smoke.py`, whose schema is
`{conversation_id: [{"turn_index": k, "messages": <prefix through user_k>}]}`
-- so each entry is already the sliced prompt prefix for its turn; we just
pick one.

Determinism note: the chop is drawn from an RNG at access time, so exact
checkpoint-resume reproducibility of *which* chop a step saw is not
guaranteed. That is an intentional trade-off for epoch-to-epoch variety;
nothing downstream depends on the chop being deterministic (the teacher
lookup is keyed by (conversation_id, turn_index), both of which are recorded
in extra_info for whatever chop was drawn).
"""

import json

import datasets
import numpy as np
import torch

from verl.utils.dataset.rl_dataset import RLHFDataset


class WildChatChopDataset(RLHFDataset):
    def _read_files_and_tokenize(self):
        # data_files points at conversations.json, not a parquet. Its top-level
        # structure is a dict keyed by conversation_id, so we parse it directly
        # rather than through datasets.load_dataset's row-oriented reader.
        conversations = {}
        for path in self.data_files:
            with open(path) as f:
                conversations.update(json.load(f))

        # The agent-loop rollout path tokenizes raw_prompt without capping length
        # (max_prompt_length is enforced only by the parent's load-time filter,
        # which we bypass). So an overlong chop would flow through and blow up the
        # rollout batch (tensor-size mismatch). Since the chop is dynamic, we can't
        # filter whole rows; instead we filter the *candidate turns* down to those
        # whose tokenized prompt fits max_prompt_length, so every draw is valid.
        # Long conversations still contribute their (shorter) early-turn chops.
        apply_kwargs = dict(**self.apply_chat_template_kwargs)
        if self.tool_schemas is not None:
            apply_kwargs["tools"] = self.tool_schemas

        def _fits(messages) -> bool:
            try:
                n = len(self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, **apply_kwargs))
            except Exception:
                return False
            return n <= self.max_prompt_length

        rows = []
        n_turns_in = n_turns_kept = n_conv_dropped = 0
        for conversation_id, turns in conversations.items():
            # Keep turns sorted by turn_index so a sampled index maps to the
            # intended prefix; drop any degenerate empty-prefix turns defensively.
            turns = sorted((t for t in turns if t.get("messages")), key=lambda t: t["turn_index"])
            n_turns_in += len(turns)
            turns = [t for t in turns if _fits(t["messages"])]
            n_turns_kept += len(turns)
            if not turns:
                n_conv_dropped += 1
                continue
            rows.append({"conversation_id": conversation_id, "turns": turns})

        self.dataframe = datasets.Dataset.from_list(rows)
        print(
            f"WildChatChopDataset: {len(self.dataframe)} conversations loaded from {self.data_files} "
            f"(kept {n_turns_kept}/{n_turns_in} candidate turns <= {self.max_prompt_length} tokens; "
            f"dropped {n_conv_dropped} conversations with no fitting turn)"
        )

        self._chop_rng = np.random.default_rng(self.seed)

    def __getitem__(self, item):
        row = self.dataframe[item]
        turns = row["turns"]
        chosen = turns[int(self._chop_rng.integers(len(turns)))]

        row_dict = {
            "data_source": "wildchat_good_smoke",
            self.prompt_key: chosen["messages"],
            "ability": "dialogue",
            "reward_model": {"style": "none", "ground_truth": ""},
            "extra_info": {
                "conversation_id": row["conversation_id"],
                "turn_index": int(chosen["turn_index"]),
            },
        }

        # Mirror RLHFDataset.__getitem__'s tail so the row matches what the
        # rollout/agent-loop expects (raw_prompt + the bookkeeping fields).
        row_dict["raw_prompt"] = self._build_messages(row_dict)
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        index = row_dict["extra_info"].get("index", 0)
        row_dict["index"] = index
        row_dict["tools_kwargs"] = row_dict["extra_info"].get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = row_dict["extra_info"].get("interaction_kwargs", {})
        return row_dict

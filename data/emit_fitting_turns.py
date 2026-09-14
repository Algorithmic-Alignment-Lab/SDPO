"""Emit the authoritative "fits max_prompt_length" turn set for a conversations.json file.

Mirrors WildChatChopDataset._read_files_and_tokenize's _fits() check EXACTLY (same tokenizer,
same apply_chat_template kwargs, same max_prompt_length) so the turn set used to size an offline
generation/judging pass (e.g. eval_judge/render_teacher_contexts.py's downstream consumers, or a
baseline/teacher completion pass over the training corpus) matches what the training dataloader
will actually be able to draw from -- rather than a hand-estimated turn count that could silently
diverge from the real filter.

Output: same {conversation_id: [{"turn_index": k, "messages": [...]}]} shape as the input, with
every non-fitting turn dropped (and every conversation with zero fitting turns dropped entirely,
same as the dataset does).
"""
import argparse
import json

from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--conversations_path", required=True)
ap.add_argument("--output_path", required=True)
ap.add_argument("--model", default="Qwen/Qwen3-32B")
ap.add_argument("--max_prompt_length", type=int, default=2048)
ap.add_argument("--enable_thinking", action="store_true", default=False)
a = ap.parse_args()

tokenizer = AutoTokenizer.from_pretrained(a.model)
apply_kwargs = {"enable_thinking": a.enable_thinking}


def fits(messages) -> bool:
    try:
        n = len(tokenizer.apply_chat_template(messages, add_generation_prompt=True, **apply_kwargs))
    except Exception:
        return False
    return n <= a.max_prompt_length


with open(a.conversations_path) as f:
    conversations = json.load(f)

out = {}
n_turns_in = n_turns_kept = n_conv_dropped = 0
for conv_id, turns in conversations.items():
    turns = sorted((t for t in turns if t.get("messages")), key=lambda t: t["turn_index"])
    n_turns_in += len(turns)
    kept = [t for t in turns if fits(t["messages"])]
    n_turns_kept += len(kept)
    if not kept:
        n_conv_dropped += 1
        continue
    out[conv_id] = kept

with open(a.output_path, "w") as f:
    json.dump(out, f)

print(
    f"emit_fitting_turns: {len(out)}/{len(conversations)} conversations kept, "
    f"{n_turns_kept}/{n_turns_in} candidate turns <= {a.max_prompt_length} tokens "
    f"(model={a.model}, enable_thinking={a.enable_thinking}) -> {a.output_path}"
)

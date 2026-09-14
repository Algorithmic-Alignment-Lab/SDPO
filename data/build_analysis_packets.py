"""Build one compact ground-truth packet per conversation for the GOOD focus
investigation. Each packet gives an analysis subagent the raw topic flow (user
turns + truncated assistant replies) plus the expected language, so it can judge
whether the tracked goals actually follow the conversation.

Usage:
    python data/build_analysis_packets.py \
        --conversations datasets/wildchat_good_diag30/conversations.json \
        --languages datasets/wildchat_good_diag30/languages.json \
        --out_dir <packets_dir>
"""

import argparse
import json
import os


def _trunc(s, n):
    s = " ".join((s or "").split())
    return s if len(s) <= n else s[:n] + " …"


def build_packet(conv_id, turns, language):
    # turns: list of {turn_index, messages}; messages = prefix through user_k.
    # The longest turn holds the fullest transcript.
    turns = sorted(turns, key=lambda t: t["turn_index"])
    full = max(turns, key=lambda t: len(t["messages"]))["messages"]
    lines = [f"# Conversation {conv_id}", f"**Expected goal language:** {language}", "",
             "## Ground-truth transcript (topic flow)", ""]
    turn = 0
    for m in full:
        role = m["role"]
        if role == "user":
            turn += 1
            lines.append(f"**[User turn {turn}]** {_trunc(m['content'], 500)}")
        else:
            lines.append(f"> *(assistant)* {_trunc(m['content'], 260)}")
        lines.append("")
    lines.append(f"_Total user turns available: {len(turns)} "
                 f"(turn_index 1..{turns[-1]['turn_index']})._")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--languages", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    convs = json.load(open(args.conversations))
    langs = json.load(open(args.languages))
    os.makedirs(args.out_dir, exist_ok=True)
    for conv_id, turns in convs.items():
        md = build_packet(conv_id, turns, langs.get(conv_id, "unknown"))
        with open(os.path.join(args.out_dir, f"packet_{conv_id}.md"), "w") as f:
            f.write(md)
    print(f"Wrote {len(convs)} packets to {args.out_dir}")


if __name__ == "__main__":
    main()

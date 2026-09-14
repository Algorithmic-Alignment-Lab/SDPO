"""System-message placements for the GOOD goal context.

ADDITIVE. This file introduces no change to `good_teacher_prompt.py`, which is live
training code: `build_teacher_messages` keeps producing exactly the user-turn fusion the
existing 8B/14B/32B checkpoints were distilled from, and remains the only thing training
calls. Everything here is for evaluation arms that place the block as a *system* message.

Why this exists. Every quantitative and qualitative number we have was produced with the
goal context fused into the final user turn (`build_teacher_messages`), because that is
what SDPO training does. GOOD's own interface (`good_goals.chat.GOODChat`, chat.py:92-102)
injects it as a system message. GOOD as shipped has therefore never been evaluated here,
and the two most arm-specific coded families -- `scaffold-addressed-as-user-speech`
(0.309-0.338 prompted, 0.000 vanilla at 8B/14B/32B) and the goal-list-shaped families --
are exactly what one predicts from putting a requirements document in the user's mouth.

Two placements, because "make it a system message" is ambiguous and the choice is
measurable:

  START  -- one system message before the whole conversation. This is what GOODChat does.
            The user's actual message is the last thing the model reads.
  LATE   -- one system message immediately before the final user turn. Preserves the
            block's recency (it is recomputed every turn, so it IS turn-specific
            information) while removing user attribution. The user's message is still last.
  END    -- one system message AFTER the final user turn (2026-09-13, for the goodpost
            quality eval). This is the LIC goodsplitend/goodpost placement — the strongest
            placement in the sharded-task sweep — and had never existed in the WildChat
            eval path. The block is the last thing the model reads. Same chat-template
            caveat as LATE: verify the served template renders a trailing system message
            in place rather than hoisting or dropping it.

Both leave the user's turns byte-identical, which the fused format did not.

Compatibility warning for LATE: a mid-conversation system message is rendered in place by
Qwen3's chat template, but this must be verified against the template actually served
before the arm is trusted -- some templates hoist or drop non-leading system messages,
which would silently turn LATE into START or into nothing. Verify with
`tokenizer.apply_chat_template` on the served model and assert the block appears between
the last assistant turn and the last user turn.
"""

from __future__ import annotations

START = "start"
LATE = "before_last_user"
END = "after_last_user"
PLACEMENTS = (START, LATE, END)


def build_system_context_messages(
    raw_prompt: list[dict],
    goal_context: str,
    placement: str = START,
) -> list[dict]:
    """Return `raw_prompt` with `goal_context` inserted as a system message.

    Args:
        raw_prompt: The message list the student saw -- conversation prefix through the
            final user turn. Not mutated; every message dict is copied, matching
            `build_teacher_messages`, because the eval builds several arms from one
            raw_prompt and shared dicts would let one arm's in-place edit corrupt another.
        goal_context: `format_goals_for_context(state)` output. Empty/falsy is meaningful,
            not an error -- turn 1 has no inferred goals yet -- and degenerates to the bare
            prompt, exactly as `build_teacher_messages` does, so the two arms stay
            comparable on those items.
        placement: START or LATE (see module docstring).

    Returns:
        A new message list. Shares no mutable state with `raw_prompt`.
    """
    if len(raw_prompt) == 0:
        raise ValueError("raw_prompt must contain at least the final user turn")
    if placement not in PLACEMENTS:
        raise ValueError(f"placement must be one of {PLACEMENTS}, got {placement!r}")

    messages = [dict(m) for m in raw_prompt]
    if not goal_context:
        return messages

    block = {"role": "system", "content": goal_context}

    if placement == END:
        messages.append(block)
        return messages

    if placement == START:
        # Merge into an existing leading system turn rather than emitting two, which some
        # chat templates drop or reorder. No eval item currently has one, but a
        # multi-turn or non-WildChat set could.
        if messages and messages[0].get("role") == "system":
            merged = dict(messages[0])
            merged["content"] = f"{merged['content']}\n\n{goal_context}"
            return [merged] + messages[1:]
        return [block] + messages

    # LATE: immediately before the final user turn. Index from the end so a trailing
    # non-user message (shouldn't happen on this data, but is cheap to tolerate) does not
    # put the block in the wrong place.
    idx = len(messages)
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            idx = i
            break
    return messages[:idx] + [block] + messages[idx:]

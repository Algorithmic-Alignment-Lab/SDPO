"""Pin the GOOD-teacher reprompt construction against the logic it was extracted from.

`build_teacher_messages` was lifted out of
`ray_trainer._maybe_build_self_distillation_batch` so the eval harness could build the
*exact* prompt the teacher saw during training. The refactor is only safe if it is
byte-identical to what training did before, so `_reference_build_teacher_message` below is
a verbatim copy of the original inlined implementation and every test asserts equality
against it.

Do not "clean up" the reference implementation. Its value is that it is a frozen copy of
the pre-refactor code; making it prettier destroys the thing it exists to check.
"""

import numpy as np
import pytest

from verl.utils.good_teacher_prompt import (
    DEFAULT_GOAL_CONTEXT_TEMPLATE,
    build_teacher_messages,
    load_goal_context_template,
)


def _reference_build_teacher_message(raw_prompt, goal_context, template):
    """VERBATIM pre-refactor logic from ray_trainer.py (do not modify)."""
    prompt_text = raw_prompt[-1]["content"]
    system_messages = raw_prompt[:-1]
    if goal_context:
        reprompt_text = template.format(prompt=prompt_text, goal_context=goal_context)
    else:
        reprompt_text = prompt_text
    return system_messages + [
        {"role": "user", "content": reprompt_text},
    ]


# A goal context shaped like real GOOD output (format_goals_for_context), including the
# markdown headers and bullet list that make the template's "\n\n" join load-bearing.
REAL_SHAPED_CONTEXT = (
    "## User Goals (for context)\n\n"
    "**Plausible concerns** (avoid violating):\n"
    "- Ensure descriptions are concise and within 2-3 sentences\n"
    "- Avoid concept overlap in thematic elements and execution\n\n"
    "**Current focus** (72% ± 9%):\n"
    "- Draft themed attraction concepts for a visitor centre\n"
)

MULTI_TURN_PROMPT = [
    {"role": "user", "content": "I need ten attraction ideas."},
    {"role": "assistant", "content": "Here are ten ideas: ..."},
    {"role": "user", "content": 'Alternative no 10. "Speedway Karts" - Board a scaled up version?'},
]


@pytest.mark.parametrize(
    "raw_prompt",
    [
        pytest.param(MULTI_TURN_PROMPT, id="multi_turn"),
        pytest.param([{"role": "user", "content": "single turn only"}], id="single_turn"),
        pytest.param(
            [{"role": "system", "content": "You are helpful."}] + MULTI_TURN_PROMPT,
            id="with_system_message",
        ),
    ],
)
@pytest.mark.parametrize(
    "goal_context",
    [
        pytest.param(REAL_SHAPED_CONTEXT, id="real_shaped_context"),
        pytest.param("short context", id="short_context"),
        # Turn 1 of a conversation has no inferred goals yet; the reprompt must degenerate
        # to the bare prompt rather than injecting an empty template. This is a real
        # training-time case, not a defensive edge case.
        pytest.param("", id="empty_context_degenerates"),
    ],
)
def test_matches_pre_refactor_reference(raw_prompt, goal_context):
    expected = _reference_build_teacher_message(
        raw_prompt, goal_context, DEFAULT_GOAL_CONTEXT_TEMPLATE
    )
    actual = build_teacher_messages(raw_prompt, goal_context, DEFAULT_GOAL_CONTEXT_TEMPLATE)
    assert actual == expected


def test_goal_context_fuses_into_final_user_turn_not_a_new_message():
    """The distinction that makes the eval valid.

    good_goals.GOODChat injects goal context as a *system* message. Training fused it into
    the final user turn. If this ever changes, the eval's `prompted` arm stops being the
    teacher the student was trained against.
    """
    out = build_teacher_messages(MULTI_TURN_PROMPT, REAL_SHAPED_CONTEXT)

    assert len(out) == len(MULTI_TURN_PROMPT), "no message added or removed"
    assert out[:-1] == MULTI_TURN_PROMPT[:-1], "conversation prefix must be untouched"
    assert out[-1]["role"] == "user"
    # The final turn carries BOTH the original user text and the goal context.
    assert MULTI_TURN_PROMPT[-1]["content"] in out[-1]["content"]
    assert REAL_SHAPED_CONTEXT in out[-1]["content"]
    assert out[-1]["content"] == f"{MULTI_TURN_PROMPT[-1]['content']}\n\n{REAL_SHAPED_CONTEXT}"
    assert not any(m["role"] == "system" and REAL_SHAPED_CONTEXT in m["content"] for m in out)


def test_does_not_mutate_or_alias_caller_input():
    """The eval generates several arms from one raw_prompt; aliasing would cross-contaminate.

    A shallow `list(raw_prompt[:-1])` passes the value-equality tests above while still
    sharing every prefix message dict with the caller -- so an in-place edit to one arm's
    messages would silently corrupt the other arms. This test is what catches that.
    """
    raw_prompt = [dict(m) for m in MULTI_TURN_PROMPT]
    before = [dict(m) for m in raw_prompt]

    out = build_teacher_messages(raw_prompt, REAL_SHAPED_CONTEXT)
    out[-1]["content"] = "mutated"
    out[0]["role"] = "mutated"

    assert raw_prompt == before, "caller's messages must be unchanged"


def test_accepts_numpy_object_array_from_non_tensor_batch():
    """DataProto.non_tensor_batch stores object arrays; training passes those straight in."""
    arr = np.empty(len(MULTI_TURN_PROMPT), dtype=object)
    for i, m in enumerate(MULTI_TURN_PROMPT):
        arr[i] = m

    out = build_teacher_messages(arr, REAL_SHAPED_CONTEXT)

    assert out == build_teacher_messages(MULTI_TURN_PROMPT, REAL_SHAPED_CONTEXT)


def test_custom_template_is_honoured():
    out = build_teacher_messages(
        MULTI_TURN_PROMPT, "CTX", template="GOALS:\n{goal_context}\n---\n{prompt}"
    )
    assert out[-1]["content"] == f"GOALS:\nCTX\n---\n{MULTI_TURN_PROMPT[-1]['content']}"


def test_empty_raw_prompt_raises():
    with pytest.raises(ValueError):
        build_teacher_messages([], "CTX")


def test_default_template_matches_shipped_config():
    """Guard against the constant drifting from the yaml/dataclass defaults it mirrors."""
    assert DEFAULT_GOAL_CONTEXT_TEMPLATE == "{prompt}\n\n{goal_context}"


class _Cfg:
    goal_context_template = "custom {prompt} {goal_context}"


def test_load_goal_context_template_prefers_run_config():
    assert load_goal_context_template(_Cfg()) == _Cfg.goal_context_template
    assert load_goal_context_template({"goal_context_template": "d {prompt} {goal_context}"}) == (
        "d {prompt} {goal_context}"
    )
    assert load_goal_context_template(None) == DEFAULT_GOAL_CONTEXT_TEMPLATE
    # A config that exists but doesn't set the field must fall back, not crash.
    assert load_goal_context_template(object()) == DEFAULT_GOAL_CONTEXT_TEMPLATE

"""Construction of the GOOD-teacher reprompt -- shared by training and evaluation.

This lives in its own dependency-light module (no torch, no ray, no verl config
machinery) for one reason: the evaluation harness has to build the *exact* prompt the
teacher saw during training, and the only safe way to guarantee that is for both call
sites to run the same code.

Why this matters more than it looks. The distillation target is "base model conditioned
on GOOD's goal context". The eval's `prompted` arm IS that teacher, and the headline
result is a comparison against it. If the eval reconstructs the prompt even slightly
differently -- goal context as a system message instead of appended to the final user
turn, say -- then the thing we compare against is not the thing the student was trained
toward, and the comparison silently measures the wrong quantity. `good_goals.GOODChat`
injects goal context as a *system* message, which is exactly this mistake waiting to
happen, so the eval must not reuse it for prompt construction.

The layout produced here (and previously inlined in
`verl/trainer/ppo/ray_trainer.py:_maybe_build_self_distillation_batch`) is:

    raw_prompt[:-1] + [{"role": "user", "content": template.format(
        prompt=raw_prompt[-1]["content"], goal_context=goal_context)}]

i.e. the conversation prefix is untouched and the goal context is fused into the final
user turn via the template -- NOT added as a new message.
"""

from __future__ import annotations  # keeps PEP 585 annotations lazy for older interpreters

# Mirrors the default in verl/trainer/config/actor/actor.yaml (`self_distillation.
# goal_context_template`) and the dataclass default in verl/workers/config/actor.py.
# Prefer reading the actual value from the config a given run used --
# `load_goal_context_template()` below -- and only fall back to this constant. The
# 8B runs (train-8b-235b-1, train-8b-lora-1) did not override it.
DEFAULT_GOAL_CONTEXT_TEMPLATE = "{prompt}\n\n{goal_context}"


def build_teacher_messages(
    raw_prompt: list[dict],
    goal_context: str,
    template: str = DEFAULT_GOAL_CONTEXT_TEMPLATE,
) -> list[dict]:
    """Build the goal-context-conditioned message list for one example.

    Args:
        raw_prompt: The full message list the student saw -- conversation prefix through
            the final user turn. Not mutated.
        goal_context: GOOD's `format_goals_for_context(state)` output for this
            (conversation, turn). An empty/falsy value is meaningful, not an error: turn 1
            of a conversation has no inferred goals yet, and the reprompt then degenerates
            to the bare prompt (a near-no-op distillation step for that sample). Handled
            identically here to keep training semantics unchanged.
        template: Format string with `{prompt}` and `{goal_context}` placeholders.

    Returns:
        A new message list: the prefix unchanged in content, with the final user turn's
        content replaced by the templated combination. Shares no mutable state with
        `raw_prompt`, so callers may mutate the result freely.
    """
    # len(), not a truthiness check: `not raw_prompt` raises "truth value of an array with
    # more than one element is ambiguous" on the numpy object arrays this also accepts.
    if len(raw_prompt) == 0:
        raise ValueError("raw_prompt must contain at least the final user turn")

    # Copy each message dict, not just the sequence. The eval builds several arms
    # (vanilla / prompted / placebo / ...) from one raw_prompt, and a shallow list copy
    # would leave every arm sharing the same prefix dicts -- so any downstream in-place
    # edit to one arm's messages would silently corrupt the others. Message values are
    # plain strings on this path (text-only WildChat), so a per-message dict() suffices.
    # Iterating rather than slicing also accepts the numpy object arrays that
    # DataProto.non_tensor_batch stores.
    prefix = [dict(m) for m in raw_prompt[:-1]]
    prompt_text = raw_prompt[-1]["content"]

    if goal_context:
        reprompt_text = template.format(prompt=prompt_text, goal_context=goal_context)
    else:
        reprompt_text = prompt_text

    return prefix + [{"role": "user", "content": reprompt_text}]


def load_goal_context_template(config=None) -> str:
    """Resolve the template a run actually used, falling back to the shipped default.

    Passing the live `self_distillation` config (training) keeps this exact; the eval
    passes the same config it reconstructs from the run's overrides so the two cannot
    drift. Record the resolved value in the eval run manifest.
    """
    if config is not None:
        template = getattr(config, "goal_context_template", None)
        if template is None and hasattr(config, "get"):
            template = config.get("goal_context_template", None)
        if template:
            return template
    return DEFAULT_GOAL_CONTEXT_TEMPLATE

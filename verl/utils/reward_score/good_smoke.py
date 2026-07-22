"""Trivial reward stub for the GOOD-teacher smoke test.

Reward is not a training signal anywhere on the sdpo self-distillation loss
path (see SDPO PR #1) -- this exists only because verl's reward-computation
plumbing expects *some* scoring function to be wired up for every
data_source. Always returns a constant score.
"""


def compute_score(data_source: str, solution_str: str, ground_truth: str, extra_info: dict = None) -> dict:
    return {"score": 0.0}

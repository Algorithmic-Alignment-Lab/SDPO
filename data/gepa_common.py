"""Shared pieces for the GEPA judge-calibration harness.

BASE_JUDGE_INSTRUCTION is a verbatim copy of the pairwise set-comparison task text in
good_goals.algorithm.rank_goal_sets. It is the *seed* GEPA optimises from, and the teacher
labels are produced with it, so GEPA's job is purely "make the student model reproduce the
teacher's judgements on this same task." KEEP THIS IN SYNC with algorithm.py — if the
deployed prompt changes, update here so the optimised instruction stays a drop-in.
"""

# The instruction half of the algorithm.py comparison prompt (everything except the
# interpolated transcript / options, which DSPy supplies as fields).
BASE_JUDGE_INSTRUCTION = (
    "Given the previous transcript between the agent and the human, return which goal "
    "list is more representative of what the human wants.\n\n"
    "Return only the option number of the more likely goal list (and the likely list "
    "should not contain anything that goes against or is dangerous to the human and "
    "their wants).\n\n"
    "Or return 3 if both goal lists are equally likely.\n"
    "Or return 4 if both goal lists are equally unlikely.\n\n"
    "If both goal lists contain things that go completely against the human's wants or "
    "previous transcript, return 4.\n"
    "If both goal lists are similarly plausible, return 3.\n\n"
    "Output ONLY a single number: 1, 2, 3, or 4."
)


def parse_choice(text: str) -> int:
    """Extract the 1/2/3/4 verdict from a model response, matching algorithm.py."""
    for char in (text or "").strip():
        if char in "1234":
            return int(char)
    return 3  # algorithm.py's default when nothing parses

"""Ray actor serving precomputed GOOD goal-context lookups.

Goal contexts are computed OFFLINE (see data/precompute_good_contexts.py)
rather than lazily during training. GOOD's live OpenRouter calls are pure
CPU/network I/O, not GPU work -- computing them lazily during training
serialized every batch's calls through a single actor, costing 10+ minutes
of expensive GPU-allocated time on pure network I/O for a single training
step. Precomputing once, in parallel across conversations, on a CPU-only
node means this actor is now a pure O(1) lookup at training time: no
OpenRouter calls, no `good_goals` dependency, no OPENROUTER_API_KEY needed
in the training job's environment at all.

Scale note: the full lookup table is loaded once into this single actor's
memory. Fine at the ~30-conversation smoke-test scale this was built for;
a full WildChat-1M version would need a different (e.g. on-disk/sharded)
lookup strategy -- a future task, not solved here.
"""

import json

import ray


@ray.remote(num_cpus=0)
class GoodStateCache:
    def __init__(self, goal_contexts_path: str):
        with open(goal_contexts_path) as f:
            self.goal_contexts: dict[str, str] = json.load(f)

    def get_goal_context(self, conversation_id: str, turn_index: int) -> str:
        return self.goal_contexts.get(f"{conversation_id}:{turn_index}", "")

    def get_cache_stats(self) -> dict:
        return {"num_entries": len(self.goal_contexts)}


def get_good_state_cache(goal_contexts_path: str, name: str = "good_state_cache"):
    return GoodStateCache.options(name=name, get_if_exists=True, lifetime="detached").remote(goal_contexts_path)

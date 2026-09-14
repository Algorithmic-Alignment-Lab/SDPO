"""Pin exactly which code and data produced an eval run.

This exists because the pieces this eval depends on are not all in a clean, pushed git
state, and some of them never will be at run time:

  * `good-goals` on the cluster is an **rsync snapshot, not a git repo**, and the
    authoritative branch (`fix/goal-set-topic-switch-tracking`) is unpushed with several
    uncommitted modifications plus an untracked `trace.py`. So git identity has to be passed
    in from the machine that has the repo, while content identity (md5) is captured here.
  * The `prompted` eval arm reproduces the training teacher, so the resolved
    `goal_context_template` and the builder that applies it are part of the run's identity.
  * The live-scaffold arm inherits two known-open GOOD issues (focus staleness on sub-topic
    detours, partial language localization -- see GOOD_focus_investigation.md). Any report
    built from these generations carries those caveats, so they are recorded inline rather
    than left to memory.

Usage (cluster, inside the container):
    python eval/capture_provenance.py \
        --eval_dataset datasets/wildchat_eval_250 \
        --good_goals_git "branch=fix/goal-set-topic-switch-tracking head=27ca2aa dirty=algorithm.py,atomic_goals.py,confidence.py,config.py,llm/openrouter.py untracked=trace.py" \
        --out eval_runs/provenance.json
"""

from __future__ import annotations  # `str | None` annotations on pre-3.10 interpreters

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone

# Files whose content defines what the eval *means*. A change to any of these changes the
# comparison being made, so they are fingerprinted individually rather than as a tree hash.
SEMANTIC_FILES = [
    "verl/utils/good_teacher_prompt.py",
    "verl/trainer/ppo/ray_trainer.py",
    "verl/utils/good_state_cache.py",
    "verl/utils/dataset/wildchat_chop_dataset.py",
    "data/precompute_good_contexts.py",
    "data/vllm_provider.py",
    "data/sample_wildchat_eval.py",
]

KNOWN_OPEN_GOOD_ISSUES = [
    "focus staleness on sub-topic detours: a new topic can reach the atomic/plausible pool "
    "without being promoted into the set-derived focus (worst observed: a hallucinated goal "
    "held #1 focus for six turns)",
    "partial language localization: fails on English-task/mixed conversations and garbles "
    "non-Latin scripts",
]


def md5(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_md5(root: str, suffix: str = ".py") -> dict:
    """Per-file md5 for a source tree. Works whether or not it is a git checkout."""
    out = {}
    for dirpath, _, filenames in os.walk(root):
        if "__pycache__" in dirpath:
            continue
        for name in sorted(filenames):
            if name.endswith(suffix):
                full = os.path.join(dirpath, name)
                out[os.path.relpath(full, root)] = md5(full)
    return dict(sorted(out.items()))


def git_describe(path: str) -> dict:
    """Best-effort git identity; returns a reason rather than raising when unavailable."""
    if not os.path.isdir(path):
        return {"available": False, "reason": f"{path} does not exist"}
    try:
        run = lambda *a: subprocess.run(
            a, cwd=path, capture_output=True, text=True, timeout=30
        ).stdout.strip()
        head = run("git", "rev-parse", "--short", "HEAD")
        if not head:
            return {"available": False, "reason": "not a git repository (rsync snapshot)"}
        return {
            "available": True,
            "head": head,
            "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": run("git", "status", "--porcelain").splitlines(),
        }
    except Exception as exc:  # noqa: BLE001 - provenance must never fail the run
        return {"available": False, "reason": repr(exc)}


def _parse_block_scalar(yaml_path: str, key: str) -> str | None:
    """Extract a `key: |-` block scalar without pyyaml.

    Hand-rolled on purpose. This string is load-bearing -- `prompted` IS the training
    teacher, so a null here would hide the most important fact in the manifest -- and
    provenance capture must not fail just because an optional dependency is missing from
    whatever interpreter happens to run it.
    """
    if not os.path.exists(yaml_path):
        return None
    with open(yaml_path) as f:
        lines = f.read().splitlines()

    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith(f"{key}:"):
            continue
        after = stripped[len(key) + 1 :].strip()
        if not after.startswith("|"):
            # Plain inline scalar, possibly quoted.
            return after.strip("'\"") or None
        key_indent = len(line) - len(line.lstrip())
        body = []
        for follow in lines[i + 1 :]:
            if not follow.strip():
                body.append("")
                continue
            if len(follow) - len(follow.lstrip()) <= key_indent:
                break
            body.append(follow)
        if not body:
            return None
        block_indent = min(
            len(b) - len(b.lstrip()) for b in body if b.strip()
        )
        text = "\n".join(b[block_indent:] if b.strip() else "" for b in body)
        # `|-` strips the single trailing newline; mirror that.
        return text.rstrip("\n")
    return None


def resolved_goal_context_template(sdpo_root: str) -> dict:
    """The template the prompted arm will use, read from the shipped module + config.

    Recorded because 'prompted' IS the training teacher: if this string differs from what
    training used, the headline comparison is against the wrong thing.
    """
    info: dict = {}

    # Load the module by file path rather than `import verl.utils...`: importing the package
    # executes verl/__init__.py, which needs `packaging` and other runtime deps that a bare
    # provenance-capture interpreter has no reason to have.
    module_path = os.path.join(sdpo_root, "verl/utils/good_teacher_prompt.py")
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location("_gtp_prov", module_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        info["module_default"] = mod.DEFAULT_GOAL_CONTEXT_TEMPLATE
    except Exception as exc:  # noqa: BLE001
        info["module_default_error"] = repr(exc)

    yaml_path = os.path.join(sdpo_root, "verl/trainer/config/actor/actor.yaml")
    info["actor_yaml"] = _parse_block_scalar(yaml_path, "goal_context_template")

    # The whole point is that these agree. Surface a mismatch loudly in the manifest rather
    # than leaving two different strings side by side for a reader to notice.
    if info.get("module_default") and info.get("actor_yaml"):
        info["module_matches_yaml"] = info["module_default"] == info["actor_yaml"]
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdpo_root", default=".")
    ap.add_argument("--good_goals_root", default="/good-goals")
    ap.add_argument(
        "--good_goals_git",
        default=None,
        help="Git identity captured on the machine that has the repo; the cluster copy is an "
        "rsync snapshot with no .git, so this cannot be derived here.",
    )
    ap.add_argument("--eval_dataset", default=None)
    ap.add_argument("--checkpoint", action="append", default=[], help="Repeatable.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    prov = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "sdpo": {
            "root": os.path.abspath(args.sdpo_root),
            "git": git_describe(args.sdpo_root),
            "semantic_file_md5": {
                p: md5(os.path.join(args.sdpo_root, p)) for p in SEMANTIC_FILES
            },
            "goal_context_template": resolved_goal_context_template(args.sdpo_root),
        },
        "good_goals": {
            "root": args.good_goals_root,
            "git": git_describe(args.good_goals_root),
            "git_identity_from_caller": args.good_goals_git,
            "src_md5": tree_md5(os.path.join(args.good_goals_root, "src"))
            if os.path.isdir(os.path.join(args.good_goals_root, "src"))
            else {},
            "known_open_issues": KNOWN_OPEN_GOOD_ISSUES,
        },
    }

    if args.eval_dataset:
        manifest_path = os.path.join(args.eval_dataset, "manifest.json")
        prov["eval_dataset"] = {
            "path": args.eval_dataset,
            "conversations_md5": md5(os.path.join(args.eval_dataset, "conversations.json")),
            "manifest": json.load(open(manifest_path)) if os.path.exists(manifest_path) else None,
        }

    if args.checkpoint:
        prov["checkpoints"] = {
            ckpt: {"exists": os.path.exists(ckpt)} for ckpt in args.checkpoint
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(prov, f, indent=2)

    missing = [p for p, h in prov["sdpo"]["semantic_file_md5"].items() if h is None]
    if missing:
        print(f"WARNING: semantic files not found (recorded as null): {missing}")
    print(f"wrote {args.out}")
    print(f"  good_goals src files fingerprinted: {len(prov['good_goals']['src_md5'])}")
    print(f"  goal_context_template: {prov['sdpo']['goal_context_template']}")


if __name__ == "__main__":
    main()

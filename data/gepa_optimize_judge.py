"""Calibrate GOOD's pairwise goal-set judge with DSPy GEPA.

Objective (deliberately simple, per design): make the STUDENT model (e.g. Qwen3-32B, the
model we actually annotate with) reproduce a more capable TEACHER model's judgements on the
exact pairwise set-comparison task from algorithm.rank_goal_sets. No guardrail / anti-
staleness reward is added here — long-running-goal preservation is engineered into the
teacher's prompt, not rewarded in the metric. GEPA reflectively rewrites the judge
instruction to maximise student==teacher agreement.

Pipeline:
  1. Load mined instances (gepa_mine_comparisons.py).
  2. Label each with the TEACHER model applying BASE_JUDGE_INSTRUCTION (cached).
  3. Baseline: STUDENT match-rate vs teacher on val (base instruction).
  4. GEPA: optimise the instruction on train, evaluated by student==teacher (feedback-rich).
  5. Report + save optimised instruction and before/after match-rate.

All three roles (student / teacher / reflection) are configurable LMs so we can run
everything self-hosted on vLLM (student=32B, teacher=235B) or point teacher/reflection at
another endpoint. See gepa_judge.sbatch for the co-located cluster run.
"""

import argparse
import json
import os
import random

import dspy

from gepa_common import BASE_JUDGE_INSTRUCTION, parse_choice


class GoalSetJudge(dspy.Signature):
    """Given the previous transcript between the agent and the human, return which goal
    list is more representative of what the human wants.

    Return 1 if goal list 1 is more likely, 2 if goal list 2 is more likely, 3 if both
    are equally likely, or 4 if both are equally unlikely (or both go against the human's
    wants). The chosen list should not contain anything dangerous to or against the human.
    Output ONLY a single number: 1, 2, 3, or 4."""

    transcript: str = dspy.InputField(desc="The conversation so far (Role: content per line).")
    option_1: str = dspy.InputField(desc="Goal list 1 (a JSON array of goal strings).")
    option_2: str = dspy.InputField(desc="Goal list 2 (a JSON array of goal strings).")
    choice: str = dspy.OutputField(desc="A single number: 1, 2, 3, or 4.")


def make_lm(model, api_base, api_key, temperature, max_tokens):
    return dspy.LM(
        f"openai/{model}", api_base=api_base, api_key=api_key or "EMPTY",
        temperature=temperature, max_tokens=max_tokens, cache=True,
    )


def teacher_label(lm, inst) -> int:
    """Run the TEACHER on one instance with the verbatim base instruction -> 1/2/3/4."""
    prompt = (
        f"{BASE_JUDGE_INSTRUCTION}\n\nThe previous transcript is:\n{inst['transcript']}\n\n"
        f"Option 1: {json.dumps(inst['set_1'], ensure_ascii=False)}\n"
        f"Option 2: {json.dumps(inst['set_2'], ensure_ascii=False)}\n"
    )
    resp = lm(prompt)
    text = resp[0] if isinstance(resp, list) else resp
    return parse_choice(text)


def to_example(inst) -> dspy.Example:
    return dspy.Example(
        transcript=inst["transcript"],
        option_1=json.dumps(inst["set_1"], ensure_ascii=False),
        option_2=json.dumps(inst["set_2"], ensure_ascii=False),
        choice=str(inst["teacher_label"]),
    ).with_inputs("transcript", "option_1", "option_2")


def metric(gold, pred, trace=None, pred_name=None, pred_trace=None):
    """Simple objective: does the student's verdict match the teacher's?

    Returns a dspy.Prediction(score, feedback) so GEPA's reflection LM gets a concrete,
    per-example signal to rewrite the instruction from. 1/2 (a clear winner) vs 3/4 (a tie)
    confusions are the interesting failure mode, so the feedback names both verdicts."""
    want = parse_choice(gold.choice)
    got = parse_choice(getattr(pred, "choice", ""))
    if got == want:
        fb = f"Correct: verdict {want}."
    else:
        names = {1: "option 1 wins", 2: "option 2 wins", 3: "equally likely (tie)",
                 4: "equally unlikely"}
        fb = (f"Wrong: the reference verdict is {want} ({names[want]}) but the model "
              f"answered {got} ({names[got]}). Re-read which goal list better matches "
              f"what the human is asking for in the latest turns of the transcript, and "
              f"reserve 3/4 for genuine ties rather than defaulting to them.")
    return dspy.Prediction(score=1.0 if got == want else 0.0, feedback=fb)


def evaluate(program, examples, num_threads) -> float:
    ev = dspy.Evaluate(devset=examples, metric=lambda g, p, *a: metric(g, p).score,
                       num_threads=num_threads, display_progress=True)
    res = ev(program)
    # dspy 2.6 returns a float (0-100); dspy 3.x returns an EvaluationResult(.score).
    score = getattr(res, "score", res)
    return score / 100.0 if score > 1.0 else score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    ap.add_argument("--out_dir", required=True)
    # student (the model we annotate with; the one being calibrated)
    ap.add_argument("--student_model", required=True)
    ap.add_argument("--student_base_url", default="http://localhost:8000/v1")
    ap.add_argument("--student_api_key", default="EMPTY")
    # teacher (produces the gold labels; a stronger model)
    ap.add_argument("--teacher_model", required=True)
    ap.add_argument("--teacher_base_url", default="http://localhost:8001/v1")
    ap.add_argument("--teacher_api_key", default="EMPTY")
    # reflection LM for GEPA (defaults to the teacher)
    ap.add_argument("--reflection_model", default=None)
    ap.add_argument("--reflection_base_url", default=None)
    ap.add_argument("--reflection_api_key", default=None)
    ap.add_argument("--max_tokens", type=int, default=1024,
                    help="Student/teacher output cap; must fit DSPy's adapter envelope "
                         "([[ ## choice ## ]] <n>), so leave headroom above a bare digit.")
    ap.add_argument("--reflection_max_tokens", type=int, default=8192)
    ap.add_argument("--n", type=int, default=0, help="Cap instances (0 = all).")
    ap.add_argument("--val_frac", type=float, default=0.3)
    ap.add_argument("--num_threads", type=int, default=16)
    ap.add_argument("--auto", default="light", choices=["light", "medium", "heavy"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = random.Random(args.seed)
    instances = json.load(open(args.instances))
    if args.n:
        rng.shuffle(instances)
        instances = instances[: args.n]

    student = make_lm(args.student_model, args.student_base_url, args.student_api_key,
                      temperature=0.0, max_tokens=args.max_tokens)
    teacher = make_lm(args.teacher_model, args.teacher_base_url, args.teacher_api_key,
                      temperature=0.0, max_tokens=args.max_tokens)

    # --- Step 2: teacher labels (cached to disk; resumable) ---
    label_cache = os.path.join(args.out_dir, "instances_labeled.json")
    if os.path.exists(label_cache):
        instances = json.load(open(label_cache))
        print(f"loaded {len(instances)} cached teacher-labeled instances")
    else:
        print(f"labeling {len(instances)} instances with teacher={args.teacher_model} ...")
        for k, inst in enumerate(instances):
            inst["teacher_label"] = teacher_label(teacher, inst)
            if (k + 1) % 25 == 0:
                print(f"  labeled {k+1}/{len(instances)}", flush=True)
        json.dump(instances, open(label_cache, "w"), ensure_ascii=False, indent=2)
    from collections import Counter
    print("teacher label distribution:", Counter(i["teacher_label"] for i in instances))

    examples = [to_example(i) for i in instances]
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_frac))
    valset, trainset = examples[:n_val], examples[n_val:]
    print(f"train={len(trainset)} val={len(valset)}")

    dspy.configure(lm=student)
    # The signature docstring is the seed instruction GEPA optimises from.
    program = dspy.Predict(GoalSetJudge)

    # --- Step 3: baseline student match-rate ---
    base_score = evaluate(program, valset, args.num_threads)
    print(f"\nBASELINE student==teacher on val: {base_score:.3f}")

    # --- Step 4: GEPA optimise ---
    refl_model = args.reflection_model or args.teacher_model
    refl_base = args.reflection_base_url or args.teacher_base_url
    refl_key = args.reflection_api_key or args.teacher_api_key
    reflection_lm = make_lm(refl_model, refl_base, refl_key,
                            temperature=1.0, max_tokens=args.reflection_max_tokens)
    gepa = dspy.GEPA(
        metric=metric, auto=args.auto, num_threads=args.num_threads,
        track_stats=True, reflection_lm=reflection_lm,
    )
    optimized = gepa.compile(program, trainset=trainset, valset=valset)

    # --- Step 5: report ---
    opt_score = evaluate(optimized, valset, args.num_threads)
    opt_instruction = optimized.signature.instructions
    print(f"\nOPTIMISED student==teacher on val: {opt_score:.3f}  (baseline {base_score:.3f}, "
          f"Δ {opt_score - base_score:+.3f})")
    with open(os.path.join(args.out_dir, "optimized_instruction.txt"), "w") as f:
        f.write(opt_instruction)
    optimized.save(os.path.join(args.out_dir, "optimized_program.json"))
    json.dump(
        {"baseline_val_match": base_score, "optimized_val_match": opt_score,
         "delta": opt_score - base_score, "student": args.student_model,
         "teacher": args.teacher_model, "reflection": refl_model,
         "n_train": len(trainset), "n_val": len(valset), "auto": args.auto},
        open(os.path.join(args.out_dir, "report.json"), "w"), indent=2)
    print(f"\nsaved optimised instruction + program + report to {args.out_dir}")
    print("\n=== OPTIMISED INSTRUCTION ===\n" + opt_instruction)


if __name__ == "__main__":
    main()

"""Stage 1e: generate one response per (conversation, turn) x arm x thinking-mode.

Arms and what distinguishes them (see the eval plan for why each exists):

    arm                served model     prompt
    ---------------    --------------   ----------------------------------------
    vanilla            vanilla          raw conversation
    prompted           vanilla          raw + REAL goal context   <- the teacher
    prompted-placebo   vanilla          raw + MISMATCHED context  <- content control
    distilled-fw       distilled-fw     raw conversation
    distilled-lora     distilled-lora   raw conversation
    distilled+context  distilled-fw     raw + REAL goal context   <- saturation check

Every prompt that includes goal context is built by `verl.utils.good_teacher_prompt.
build_teacher_messages`, the same function the training path uses. That is deliberate and
load-bearing: the `prompted` arm IS the teacher the student was distilled toward, so
reconstructing the prompt independently here would risk silently comparing against something
the student never saw.

Both thinking modes are generated because the distilled checkpoints were trained
**thinking-off only** -- so the thinking-on column is a clean generalization test, and the
`<think>` traces it produces feed the goal-rediscovery measurement. Sampling parameters differ
per mode (Qwen3's own recommendations) and are held identical across arms *within* a mode.

Resumable by design: output is newline-delimited JSON and already-present keys are skipped, so
a preempted run is re-launched rather than restarted.

Usage:
    python eval/generate_single_turn.py \
        --conversations datasets/wildchat_eval_250/conversations.json \
        --goal_contexts datasets/wildchat_eval_250/goal_contexts_235b.json \
        --endpoints eval_runs/endpoints.json \
        --out eval_runs/generations.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import statistics
import sys
import time

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verl.utils.good_teacher_prompt import (  # noqa: E402
    DEFAULT_GOAL_CONTEXT_TEMPLATE,
    build_teacher_messages,
)
from verl.utils.good_system_prompt import (  # noqa: E402
    END,
    LATE,
    START,
    build_system_context_messages,
)

# arm -> (served model name,
#         which goal context the prompt carries: None | "real" | "placebo" | "focus",
#         where it goes: "user" (fused into the final user turn, what training did)
#                      | "sys-start" (system message before the conversation, = GOODChat)
#                      | "sys-late"  (system message immediately before the final user turn)
#                      | "sys-end"   (system message AFTER the final user turn — the LIC
#                                     goodsplitend/goodpost placement; 2026-09-13)
#
# Placement is a THIRD field rather than more ctx_kind values because content and position
# are orthogonal: any of real/placebo/focus can be tested in any of the three positions,
# and collapsing them into one enum would need 9 names.
ARMS = {
    "vanilla": ("vanilla", None, "user"),
    "prompted": ("vanilla", "real", "user"),
    "prompted-placebo": ("vanilla", "placebo", "user"),
    "prompted-focus": ("vanilla", "focus", "user"),
    "distilled-fw": ("distilled-fw", None, "user"),
    "distilled-lora": ("distilled-lora", None, "user"),
    "distilled+context": ("distilled-fw", "real", "user"),
    # --- placement arms (2026-08-11). Every number we have was produced with the block
    # fused into the user turn; GOOD as shipped puts it in a system message and has never
    # been evaluated here. `prompted-sys` IS GOOD as shipped and is the missing control.
    "prompted-sys": ("vanilla", "real", "sys-start"),
    "prompted-sys-late": ("vanilla", "real", "sys-late"),
    "prompted-focus-sys": ("vanilla", "focus", "sys-start"),
    # sys-end quality eval (2026-09-13): the context FILE decides the content (flat vs
    # goodpost blocks) — run the script once per contexts file with this arm.
    "prompted-sys-end": ("vanilla", "real", "sys-end"),
}

PLACEMENTS = {"user", "sys-start", "sys-late", "sys-end"}

# The placement arms are opt-in: they must be named explicitly in --arms. Leaving them out
# of the default keeps an unmodified command line generating exactly the same seven arms and
# the same row count as every run so far.
DEFAULT_ARMS = ("vanilla", "prompted", "prompted-placebo", "prompted-focus",
                "distilled-fw", "distilled-lora", "distilled+context")


def focus_only(context: str) -> str:
    """Keep the block's header + provenance preamble and the focus section; drop the rest.

    ABLATION ARM. Measured over all 1474 eval contexts, the injected block is a median of 45
    background goals plus 6 focus goals, and the background section is **85-86% of the
    characters**. The pool was also filled to a quota -- the old proposer was asked for
    `max(5, 1.2*max_plausible_goals - current)` goals, i.e. 54 from a single first user turn --
    so much of it was speculation generated to hit a number rather than inference from evidence.

    The goal SETS (the focus) are what is supposed to drive the response. This arm isolates them,
    so the eval can distinguish:
      * focus-only >= vanilla  -> the background block is the cost; fix is prompt-side filtering
      * focus-only ~ full prompted, both < vanilla -> the focus itself does not help
      * focus-only < both -> injecting context at all is costly regardless of content
    Without this arm, "the background goals are the problem" is an inference from text ratios,
    not a measurement.

    REWRITTEN 2026-08-11 for the redesigned `format_goals_for_context`. The old parser walked
    forward looking for a line starting `**Current focus` and kept only that section, dropping
    everything before it that was not a `## ` header. Under the new block the focus section is
    FIRST and is preceded by a provenance/how-to-use preamble that is plain prose -- the old
    parser would have deleted the preamble and produced an unframed focus list, i.e. silently
    confounded the focus-only arm with the framing change. It now keeps the prefix up to the
    background marker instead, which is correct for the new layout and still yields the old
    behaviour's intent (header + focus, no background).

    Recognises both layouts so a focus-only arm can still be built from the ALREADY ANNOTATED
    `goal_contexts_235b.json` (old marker) without re-running the 235B annotation.
    """
    if not context:
        return context

    lines = context.splitlines()
    new_bg = "**Other things that may still be true of this user"
    old_bg = "**Plausible concerns"

    # New layout: focus first. Keep everything up to the background section.
    if any(ln.strip().startswith("**Most likely right now") for ln in lines):
        out = []
        for ln in lines:
            if ln.strip().startswith(new_bg):
                break
            out.append(ln)
        return "\n".join(out).strip()

    # Old layout: background first, focus last. Keep `## ` headers + the focus section.
    out, keep = [], False
    for ln in lines:
        s = ln.strip()
        if s.startswith("## "):
            out.append(ln)
            continue
        if s.startswith("**Current focus"):
            keep = True
        elif s.startswith(old_bg) or (s.startswith("**") and keep):
            keep = False
        if keep:
            out.append(ln)
    return "\n".join(out).strip()

# Qwen3's documented sampling recommendations differ by mode. Held identical across arms
# within a mode so no arm gets a decoding advantage.
MODE_PARAMS = {
    "off": {"temperature": 0.7, "top_p": 0.8, "top_k": 20},
    "on": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
}


def stable_int(*parts: str) -> int:
    """Deterministic integer from strings.

    hashlib, NOT the builtin hash(): hash() on str is salted per process, so using it would
    make seeds and placebo-donor choices differ between runs and destroy reproducibility.
    """
    digest = hashlib.md5("|".join(parts).encode()).hexdigest()
    return int(digest[:8], 16)


def split_think(text: str) -> tuple[str, str, bool]:
    """Split a Qwen3 response into (think, answer, think_unclosed).

    vLLM returns the reasoning block inline in message.content unless a reasoning parser is
    configured, so this owns the parse. An unclosed <think> means the trace was cut off by
    max_tokens -- reported rather than silently treated as an empty answer, because a
    truncation rate that differs across arms invalidates the comparison.
    """
    if "<think>" not in text and "</think>" not in text:
        return "", text, False
    body = text.split("<think>", 1)[-1] if "<think>" in text else text
    if "</think>" in body:
        think, answer = body.split("</think>", 1)
        return think.strip(), answer.strip(), False
    return body.strip(), "", True


def assign_placebo_donors(contexts: dict[str, str]) -> dict[str, str]:
    """Map each turn key to a length-matched goal context from a DIFFERENT conversation.

    The placebo arm answers "is GOOD's context *content* load-bearing, or does any block of
    plausible goal-ish text change behaviour?" For that to be a fair control the donor must
    match the real context in size -- otherwise the arms differ in prompt length as well as
    content, and length alone could explain any gap.

    Deterministic: candidates are ranked by |length difference| then key, and the choice among
    the closest few is rotated by a stable hash so a handful of donors do not serve every
    target. Returns {turn_key: donor_key}.
    """
    by_len = sorted(((len(v), k) for k, v in contexts.items() if v.strip()))
    donors: dict[str, str] = {}
    n_candidates = 5

    for key, ctx in contexts.items():
        if not ctx.strip():
            continue
        target_conv = key.rsplit(":", 1)[0]
        target_len = len(ctx)
        ranked = sorted(
            (
                (abs(length - target_len), donor_key)
                for length, donor_key in by_len
                if donor_key.rsplit(":", 1)[0] != target_conv
            )
        )
        if not ranked:
            continue
        pool = ranked[:n_candidates]
        donors[key] = pool[stable_int("placebo", key) % len(pool)][1]
    return donors


def build_work(conversations: dict, contexts: dict, donors: dict, arms, modes,
               template: str, extra_sample_arms: set[str]) -> list[dict]:
    work = []
    for conv_id, turns in conversations.items():
        for turn in turns:
            turn_index = turn["turn_index"]
            key = f"{conv_id}:{turn_index}"
            real_ctx = contexts.get(key, "")
            placebo_key = donors.get(key)
            placebo_ctx = contexts.get(placebo_key, "") if placebo_key else ""

            for arm in arms:
                model, ctx_kind, placement = ARMS[arm]
                if ctx_kind == "real":
                    ctx, donor = real_ctx, None
                elif ctx_kind == "focus":
                    ctx, donor = focus_only(real_ctx), None
                    if not ctx:
                        continue   # no focus section -> not a valid focus-only item
                elif ctx_kind == "placebo":
                    ctx, donor = placebo_ctx, placebo_key
                    # No donor available (e.g. a single-conversation debug set) -- skip rather
                    # than silently emit a bare-prompt row that looks like a placebo.
                    if not ctx:
                        continue
                else:
                    ctx, donor = "", None

                if not ctx_kind:
                    messages = [dict(m) for m in turn["messages"]]
                elif placement == "user":
                    messages = build_teacher_messages(turn["messages"], ctx, template)
                else:
                    messages = build_system_context_messages(
                        turn["messages"], ctx,
                        {"sys-start": START, "sys-late": LATE, "sys-end": END}[placement],
                    )

                n_samples = 2 if arm in extra_sample_arms else 1
                for mode in modes:
                    for sample in range(n_samples):
                        work.append({
                            "conversation_id": conv_id,
                            "turn_index": turn_index,
                            "arm": arm,
                            "mode": mode,
                            "sample": sample,
                            "model": model,
                            "messages": messages,
                            "goal_context_present": bool(ctx),
                            "goal_context_chars": len(ctx),
                            "goal_context_placement": placement if ctx_kind else None,
                            "placebo_donor_key": donor,
                        })
    return work


def row_key(row: dict) -> str:
    return f"{row['conversation_id']}:{row['turn_index']}:{row['arm']}:{row['mode']}:{row['sample']}"


async def generate_one(client: httpx.AsyncClient, item: dict, endpoints: dict,
                       max_tokens: dict[str, int], sem: asyncio.Semaphore,
                       timeout: float, max_model_len: int | None = None) -> dict:
    mode = item["mode"]
    params = dict(MODE_PARAMS[mode])
    top_k = params.pop("top_k")
    base_url = endpoints[item["model"]].rstrip("/")

    payload = {
        "model": item["model"],
        "messages": item["messages"],
        "max_tokens": max_tokens[mode],
        # Seeded per (arm, mode, item, sample) so reruns reproduce and the vanilla-vs-vanilla
        # negative control gets genuinely different draws rather than the same text twice.
        "seed": stable_int(item["arm"], mode, item["conversation_id"],
                           str(item["turn_index"]), str(item["sample"])),
        # Qwen3 ships thinking ON by default; this flag is the entire mode manipulation.
        "chat_template_kwargs": {"enable_thinking": mode == "on"},
        **params,
    }
    payload["top_k"] = top_k  # vLLM accepts top_k as an OpenAI-API extension

    # NOTE: this allowlist is the ONLY path from the work item to the output row. A field added
    # to the work dict but not listed here is silently dropped -- which is exactly what happened
    # to `goal_context_placement` on its first run: the judge then saw a null placement, assumed
    # the default "user" fusion, and stripped goal_context_chars off the end of real user turns,
    # handing judges an empty question. Add new fields in BOTH places.
    out = {k: item[k] for k in (
        "conversation_id", "turn_index", "arm", "mode", "sample", "model",
        "goal_context_present", "goal_context_chars", "goal_context_placement",
        "placebo_donor_key")}
    out["prompt_messages"] = item["messages"]

    async with sem:
        start = time.monotonic()
        try:
            resp = await client.post(f"{base_url}/chat/completions", json=payload,
                                     timeout=timeout)
            # prompt_tokens + max_tokens must fit the server's --max-model-len. That squeeze is
            # ARM-DEPENDENT: context-bearing arms carry an extra 2-3k tokens of goal context, so
            # they alone get rejected on the longest turns, producing structured (not random)
            # missing data exactly where prompts are biggest. The server's error states the real
            # token counts, so use them to retry with the largest budget that actually fits rather
            # than guessing a cap or silently dropping the row.
            if resp.status_code == 400 and "maximum context length" in resp.text:
                m = re.search(r"(\d+) in the messages", resp.text)
                if m and max_model_len:
                    room = max_model_len - int(m.group(1)) - 64  # 64 = template/BOS slack
                    if room >= 256:
                        out["max_tokens_clamped_from"] = payload["max_tokens"]
                        out["max_tokens_clamped_to"] = room
                        payload["max_tokens"] = room
                        resp = await client.post(f"{base_url}/chat/completions", json=payload,
                                                 timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:  # noqa: BLE001 - one failure must not kill the sweep
            out["error"] = repr(exc)[:400]
            out["latency_s"] = round(time.monotonic() - start, 3)
            return out
        latency = time.monotonic() - start

    choice = data["choices"][0]
    content = choice["message"]["content"] or ""
    think, answer, think_unclosed = split_think(content)
    usage = data.get("usage", {})

    out.update({
        "raw_content": content,
        "think": think,
        "answer": answer,
        "think_unclosed": think_unclosed,
        "finish_reason": choice.get("finish_reason"),
        "truncated": choice.get("finish_reason") == "length",
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "latency_s": round(latency, 3),
        "seed": payload["seed"],
        "sampling": {"max_tokens": max_tokens[mode], "top_k": top_k, **params},
    })
    return out


async def run(work: list[dict], endpoints: dict, out_path: str, concurrency: int,
              max_tokens: dict[str, int], timeout: float,
              max_model_len: int | None = None) -> None:
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()
    done = 0
    total = len(work)

    limits = httpx.Limits(max_connections=concurrency, max_keepalive_connections=concurrency)
    async with httpx.AsyncClient(limits=limits) as client:
        with open(out_path, "a") as fh:

            async def worker(item):
                nonlocal done
                row = await generate_one(client, item, endpoints, max_tokens, sem, timeout,
                                         max_model_len)
                async with write_lock:
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fh.flush()  # flush per row: a preempted job must not lose buffered work
                    done += 1
                    if done % 200 == 0 or done == total:
                        print(f"  {done}/{total}", flush=True)

            await asyncio.gather(*(worker(i) for i in work))


def summarize(out_path: str) -> None:
    """Per (arm, mode) health numbers -- verification #3 (truncation) and #9 (length audit).

    Truncation and length are reported per arm because a *difference* between arms is what
    invalidates the judging, not the absolute level.
    """
    groups: dict[tuple, list] = {}
    errors: dict[tuple, int] = {}
    for line in open(out_path):
        row = json.loads(line)
        gk = (row["arm"], row["mode"])
        if row.get("error"):
            errors[gk] = errors.get(gk, 0) + 1
            continue
        groups.setdefault(gk, []).append(row)

    print(f"\n{'arm':<19} {'mode':<5} {'n':>6} {'trunc%':>7} {'unclosed%':>10} "
          f"{'compl_p50':>10} {'ans_chars':>10} {'lat_p50':>8} {'err':>5}")
    for gk in sorted(set(groups) | set(errors)):
        rows = groups.get(gk, [])
        if not rows:
            print(f"{gk[0]:<19} {gk[1]:<5} {0:>6} {'-':>7} {'-':>10} {'-':>10} {'-':>10} "
                  f"{'-':>8} {errors.get(gk, 0):>5}")
            continue
        trunc = 100 * sum(r["truncated"] for r in rows) / len(rows)
        unclosed = 100 * sum(r["think_unclosed"] for r in rows) / len(rows)
        compl = [r["completion_tokens"] for r in rows if r.get("completion_tokens")]
        print(f"{gk[0]:<19} {gk[1]:<5} {len(rows):>6} {trunc:>7.1f} {unclosed:>10.1f} "
              f"{int(statistics.median(compl)) if compl else 0:>10} "
              f"{int(statistics.median([len(r['answer']) for r in rows])):>10} "
              f"{statistics.median([r['latency_s'] for r in rows]):>8.2f} "
              f"{errors.get(gk, 0):>5}")

    for gk, rows in sorted(groups.items()):
        t = 100 * sum(r["truncated"] for r in rows) / len(rows)
        if t > 5:
            print(f"\nWARNING: {gk[0]}/{gk[1]} truncated {t:.1f}% of responses. If this rate "
                  f"differs materially across arms, raise --max_tokens_{gk[1]} and regenerate "
                  f"before judging -- a length confound biases every judge.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--goal_contexts", required=True)
    ap.add_argument("--endpoints", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument("--extra_arms", default="",
                    help="Declare additional arms as name:served_model:ctx_kind[:placement], "
                         "comma-separated. ctx_kind is one of none|real|placebo|focus; optional "
                         "placement is one of user|sys-start|sys-late and defaults to user (the "
                         "training format). Lets a checkpoint ladder "
                         "(e.g. lora144:lora144:none,lora200:lora200:none) be evaluated without "
                         "editing this file. Served model names must exist in endpoints.json.")
    ap.add_argument("--modes", default="off,on")
    ap.add_argument("--extra_sample_arms", default="vanilla",
                    help="Arms generated twice at different seeds, for the vanilla-vs-vanilla "
                         "negative control. Comma-separated.")
    ap.add_argument("--max_tokens_off", type=int, default=1024)
    ap.add_argument("--max_tokens_on", type=int, default=4096,
                    help="Thinking traces are long; set this from the 0.3 calibration run.")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--limit_conversations", type=int, default=None,
                    help="Debug/smoke: only the first N conversations (sorted for determinism).")
    ap.add_argument("--summarize_only", action="store_true")
    args = ap.parse_args()

    if args.summarize_only:
        summarize(args.out)
        return

    # Register CLI-declared arms before validating, so a checkpoint ladder needs no code change.
    for spec in (a for a in args.extra_arms.split(",") if a):
        parts = spec.split(":")
        if len(parts) not in (3, 4):
            sys.exit("--extra_arms entry must be name:served_model:ctx_kind[:placement], "
                     f"got {spec!r}")
        name, model, ctx = parts[:3]
        placement = parts[3] if len(parts) == 4 else "user"
        if ctx not in ("none", "real", "placebo", "focus"):
            sys.exit(f"bad ctx_kind {ctx!r} in {spec!r}; use none|real|placebo|focus")
        if placement not in PLACEMENTS:
            sys.exit(f"bad placement {placement!r} in {spec!r}; use {sorted(PLACEMENTS)}")
        ARMS[name] = (model, None if ctx == "none" else ctx, placement)

    arms = [a for a in args.arms.split(",") if a]
    unknown = set(arms) - set(ARMS)
    if unknown:
        sys.exit(f"unknown arms: {sorted(unknown)}; known: {sorted(ARMS)}")
    modes = [m for m in args.modes.split(",") if m]
    if set(modes) - set(MODE_PARAMS):
        sys.exit(f"unknown modes: {sorted(set(modes) - set(MODE_PARAMS))}")

    conversations = json.load(open(args.conversations))
    if args.limit_conversations:
        keep = sorted(conversations)[: args.limit_conversations]
        conversations = {k: conversations[k] for k in keep}
    contexts = json.load(open(args.goal_contexts))

    endpoints_blob = json.load(open(args.endpoints))
    endpoints = endpoints_blob["model_endpoints"]
    needed = {ARMS[a][0] for a in arms}
    missing = needed - set(endpoints)
    if missing:
        sys.exit(f"endpoints.json has no entry for served model(s): {sorted(missing)}")

    # Fail loudly if the goal contexts do not cover this eval set. A silently-missing context
    # makes `prompted` degenerate to `vanilla` for that turn, which would quietly dilute the
    # single most important comparison in the eval.
    turn_keys = {f"{c}:{t['turn_index']}" for c, ts in conversations.items() for t in ts}
    covered = sum(1 for k in turn_keys if contexts.get(k, "").strip())
    print(f"goal-context coverage: {covered}/{len(turn_keys)} turns")
    if covered < len(turn_keys):
        print(f"  NOTE: {len(turn_keys) - covered} turn(s) have no/empty context; for those, "
              f"context-bearing arms degenerate to the bare prompt exactly as training did.")

    donors = assign_placebo_donors({k: contexts.get(k, "") for k in turn_keys})
    print(f"placebo donors assigned for {len(donors)} turns")

    work = build_work(conversations, contexts, donors, arms, modes,
                      DEFAULT_GOAL_CONTEXT_TEMPLATE, set(filter(None, args.extra_sample_arms.split(","))))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    # Only SUCCESSFUL rows count as done. Error rows are written to the output for diagnosis, but
    # counting them as complete meant a failure could never be retried -- so a fixable cause (here,
    # prompt+max_tokens overflowing --max-model-len on long context-bearing prompts) would be
    # permanently baked into the dataset as missing data.
    already = set()
    n_failed_rows = 0
    if os.path.exists(args.out):
        for line in open(args.out):
            try:
                row = json.loads(line)
            except Exception:  # noqa: BLE001 - a torn final line from a kill is expected
                continue
            if row.get("error"):
                n_failed_rows += 1
                continue
            already.add(row_key(row))
    if n_failed_rows:
        print(f"note: {n_failed_rows} previously-errored row(s) will be RETRIED")
    todo = [w for w in work if row_key(w) not in already]
    print(f"work: {len(work)} total, {len(already)} already done, {len(todo)} to generate")

    if todo:
        max_tokens = {"off": args.max_tokens_off, "on": args.max_tokens_on}
        start = time.monotonic()
        asyncio.run(run(todo, endpoints, args.out, args.concurrency, max_tokens, args.timeout,
                        endpoints_blob.get("max_model_len")))
        print(f"generated {len(todo)} rows in {time.monotonic() - start:.0f}s")

    summarize(args.out)


if __name__ == "__main__":
    main()

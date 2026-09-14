"""Tests for the shared item-rendering/leak-scan path. Runnable directly (no pytest needed):

    python SDPO/tests/eval/test_item_rendering.py
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval"))
import item_rendering as ir  # noqa: E402

CTX = ("Inferred notes about this user\n- wants a diagram\n- prefers terse answers\n"
       "- is migrating a Flask app to FastAPI and cares about backwards compatibility "
       "of the /v1 routes during the cutover window")


def row_user_fused():
    return {
        "prompt_messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "final question?\n\n" + CTX},
        ],
        "goal_context_chars": len(CTX),
        "goal_context_placement": "user",
    }


def row_sys_placed():
    return {
        "prompt_messages": [
            {"role": "system", "content": CTX},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "final question?"},
        ],
        "goal_context_chars": len(CTX),
        "goal_context_placement": "sys-start",
    }


def row_legacy_no_placement():
    r = row_user_fused()
    del r["goal_context_placement"]  # pre-2026-08-11 rows: must default to "user"
    return r


def row_vanilla():
    return {
        "prompt_messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "final question?"},
        ],
        "goal_context_chars": 0,
        "goal_context_placement": None,
    }


def test_stripping_every_placement():
    for name, row in [("user", row_user_fused()), ("sys", row_sys_placed()),
                      ("legacy", row_legacy_no_placement()), ("vanilla", row_vanilla())]:
        text = ir.render_markdown_conversation(row)
        assert "Inferred notes" not in text, f"{name}: marker survived stripping"
        assert "final question?" in text, f"{name}: real user question was amputated"
        assert not ir.scan_text_for_leaks(text, CTX), f"{name}: leak scan flagged clean render"


def test_scan_catches_static_marker():
    assert ir.scan_text_for_leaks("blah Inferred notes about this user blah")


def test_scan_catches_context_probe_without_marker():
    # A future template rewrite could drop every static marker; the substring probes from the
    # actual context must still catch the leak.
    reworded = CTX.replace("Inferred notes about this user", "Background signals")
    assert ir.scan_text_for_leaks("conversation...\n" + reworded, goal_context=reworded)


def test_assert_items_clean_hard_fails_on_planted_leak():
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "item_000_o0.md"), "w").write("clean conversation")
        open(os.path.join(d, "item_001_o0.md"), "w").write("oops\n" + CTX)
        try:
            ir.assert_items_clean(d, {"item_001_o0": CTX})
        except RuntimeError as e:
            assert "item_001_o0" in str(e)
        else:
            raise AssertionError("planted leak was not caught")
        # and a clean dir passes
        os.remove(os.path.join(d, "item_001_o0.md"))
        assert ir.assert_items_clean(d) == 1


def test_judge_protocol_uses_shared_functions():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "eval"))
    import judge_protocol as jp
    assert jp._judged_prefix is ir.judged_prefix
    assert jp._final_user_turn is ir.final_user_turn


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")

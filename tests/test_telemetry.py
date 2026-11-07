import sys
from pathlib import Path

import pytest

# Add project root to path so imports work cleanly
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from sandbox.telemetry import inject_telemetry_wrapper

IMPORT_LINE = "from telemetry_tracker import attach_tensor_tracer"
TRACER_CALL = "attach_tensor_tracer(model)"

BASIC_SCRIPT = (
    "import torch\n"
    "class Model(torch.nn.Module): ...\n"
    "model = Model()\n"
    "y = model(x)"
)


def _lines(s: str) -> list[str]:
    return s.splitlines()


def test_import_is_prepended_as_first_line():
    out = _lines(inject_telemetry_wrapper(BASIC_SCRIPT))
    assert out[0] == IMPORT_LINE


def test_tracer_call_sits_directly_above_forward_call():
    """Order matters: the hook must be attached BEFORE the forward pass runs."""
    out = _lines(inject_telemetry_wrapper(BASIC_SCRIPT))
    i = out.index("y = model(x)")
    assert out[i - 1] == TRACER_CALL


def test_original_lines_are_preserved_in_order():
    out = _lines(inject_telemetry_wrapper(BASIC_SCRIPT))
    kept = [ln for ln in out if ln not in (IMPORT_LINE, TRACER_CALL)]
    assert kept == _lines(BASIC_SCRIPT)


def test_indentation_is_copied_from_forward_call_spaces():
    script = (
        "import torch\n"
        "def run_test():\n"
        "    with torch.no_grad():\n"
        "        output = model(x)"
    )
    out = _lines(inject_telemetry_wrapper(script))
    i = out.index("        output = model(x)")
    assert out[i - 1] == "        " + TRACER_CALL


def test_indentation_is_copied_from_forward_call_tabs():
    out = _lines(inject_telemetry_wrapper("def f():\n\ty = model(x)"))
    i = out.index("\ty = model(x)")
    assert out[i - 1] == "\t" + TRACER_CALL


def test_output_is_valid_python_for_indented_forward_call():
    script = (
        "def run_test():\n"
        "    with open('f') as fh:\n"
        "        y = model(x)\n"
    )
    compile(inject_telemetry_wrapper(script), "<instrumented>", "exec")


@pytest.mark.parametrize(
    "call_line",
    ["y = model(x)", "out = model(inputs)", "output = model(x, mask)", "return model(x)", "model(x)"],
)
def test_matches_common_forward_call_shapes(call_line):
    out = _lines(inject_telemetry_wrapper(f"import torch\n{call_line}"))
    i = out.index(call_line)
    assert out[i - 1] == TRACER_CALL


@pytest.mark.parametrize(
    "decoy",
    ["z = submodel(x)", "z = self.model(x)", "# model(x)", "def model(x):", "model = Model()"],
)
def test_decoy_lines_do_not_match_alone(decoy):
    """A decoy with no real forward call must raise, not silently instrument."""
    with pytest.raises(ValueError):
        inject_telemetry_wrapper(f"import torch\n{decoy}")


def test_idempotent_second_call_changes_nothing():
    once = inject_telemetry_wrapper(BASIC_SCRIPT)
    assert inject_telemetry_wrapper(once) == once


def test_hook_attached_only_once_for_multiple_forward_calls():
    out = inject_telemetry_wrapper("import torch\ny = model(x)\nz = model(x)")
    assert out.count(TRACER_CALL) == 1
    assert _lines(out).index(TRACER_CALL) < _lines(out).index("y = model(x)")


def test_no_forward_call_raises_value_error():
    with pytest.raises(ValueError):
        inject_telemetry_wrapper("import torch\nx = 1")


def test_future_import_stays_first():
    script = "from __future__ import annotations\nimport torch\ny = model(x)"
    out = _lines(inject_telemetry_wrapper(script))
    assert out[0] == "from __future__ import annotations"
    assert out[1] == IMPORT_LINE
    compile(inject_telemetry_wrapper(script), "<instrumented>", "exec")


def test_trailing_newline_is_preserved():
    assert inject_telemetry_wrapper(BASIC_SCRIPT + "\n").endswith("\n")
    assert not inject_telemetry_wrapper(BASIC_SCRIPT).endswith("\n")


def test_no_trailing_space_after_tracer_call():
    out = _lines(inject_telemetry_wrapper(BASIC_SCRIPT))
    assert TRACER_CALL in out  # exact match, so a trailing space would fail


def test_input_string_is_not_mutated():
    original = BASIC_SCRIPT
    inject_telemetry_wrapper(original)
    assert original == BASIC_SCRIPT

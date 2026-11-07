"""
Docker-free tests for the sandbox harness (pure string work + local python exec).
Why: the harness is the only thing between Coder output and the container. If it does not
compile, or reports success on failure, the whole Critic loop is lying to itself.
"""
import subprocess
import sys
import textwrap

import pytest

from sandbox.docker_controller import DockerSandboxController
from sandbox.telemetry import inject_telemetry_wrapper

MODEL = textwrap.dedent("""\
    import torch.nn as nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)
""")


@pytest.fixture
def controller():
    # bypass __init__: it talks to the Docker daemon
    return DockerSandboxController.__new__(DockerSandboxController)


def test_harness_compiles(controller):
    h = controller._generate_execution_harness(MODEL, {"x": "(2, 4)"})
    compile(h, "main_run.py", "exec")


def test_harness_calls_cuda_with_parens(controller):
    h = controller._generate_execution_harness(MODEL, {"x": "(2, 4)"})
    assert "torch.randn(2, 4).cuda()" in h


def test_multiple_inputs_all_defined(controller):
    h = controller._generate_execution_harness(MODEL, {"x": "(2, 4)", "y": "(2, 4)"})
    compile(h, "main_run.py", "exec")
    assert "y=torch.randn(2, 4).cuda()" in h


def test_injection_lands_before_forward_call(controller):
    h = inject_telemetry_wrapper(controller._generate_execution_harness(MODEL, {"x": "(2, 4)"}))
    lines = h.splitlines()
    i = next(n for n, l in enumerate(lines) if l.strip() == "attach_tensor_tracer(model)")
    assert "output=model(" in lines[i + 1]
    import re
    assert sum(bool(re.search(r"(?:^|=|return)\s*model\(", l)) for l in lines) == 1   # single call site
    compile(h, "main_run.py", "exec")


def test_failure_exits_nonzero_and_prints_marker_we_check_for(tmp_path, controller):
    """Regression: harness printed 'execution error' but exited 0 and the controller grepped
    for 'execute error', so failures passed silently. No GPU needed: the no-Module path
    raises before any .cuda() on the model, and inputs fail first on CPU-only hosts anyway."""
    h = controller._generate_execution_harness("x = 1", {"x": "(2, 4)"})
    # stub the tracker module so the script can start without torch.cuda
    (tmp_path / "telemetry_tracker.py").write_text(
        "shape_log = {}\ndef attach_tensor_tracer(m):\n    pass\n"
    )
    (tmp_path / "main_run.py").write_text(h.replace(".cuda()", ""))
    p = subprocess.run([sys.executable, "main_run.py"], cwd=tmp_path, capture_output=True, text=True)
    out = (p.stdout + p.stderr).lower()
    # torch/jax may be absent locally; either way the process must NOT exit 0
    assert p.returncode != 0
    if "no nn.module class found" in out:
        assert "execution error" in out


# ---- multi-input + correct model selection (executed for real on CPU) ----------------------
def _run_harness(tmp_path, controller, code, inputs):
    h = controller._generate_execution_harness(code, inputs).replace(".cuda()", "")
    (tmp_path / "telemetry_tracker.py").write_text("shape_log = {}\ndef attach_tensor_tracer(m):\n    pass\n")
    (tmp_path / "jax.py").write_text("")          # harness imports jax; stub so CPU-only dev boxes pass
    (tmp_path / "main_run.py").write_text(h)
    p = subprocess.run([sys.executable, "main_run.py"], cwd=tmp_path, capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


MULTI = textwrap.dedent("""\
    import torch
    import torch.nn as nn
    from torch.nn import Linear

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = Linear(4, 4)
        def forward(self, x):
            return self.fc(x)

    class Top(nn.Module):
        def __init__(self):
            super().__init__()
            self.b = Block()
        def forward(self, x, mask):
            return self.b(x) * mask
""")


def test_harness_feeds_all_inputs_and_picks_top_level_class(tmp_path, controller):
    pytest.importorskip("torch")
    rc, out = _run_harness(tmp_path, controller, MULTI, {"x": "(2, 4)", "mask": "(2, 4)"})
    assert rc == 0 and "Executing forward pass success" in out, out


def test_harness_positional_fallback_when_names_do_not_match(tmp_path, controller):
    pytest.importorskip("torch")
    code = MULTI.replace("def forward(self, x, mask)", "def forward(self, a, b)").replace("self.b(x) * mask", "self.b(a) * b")
    rc, out = _run_harness(tmp_path, controller, code, {"x": "(2, 4)", "mask": "(2, 4)"})
    assert rc == 0, out


def test_imported_module_class_is_not_mistaken_for_the_model(tmp_path, controller):
    pytest.importorskip("torch")
    code = "import torch.nn as nn\nfrom torch.nn import Linear\n"   # nothing defined here
    rc, out = _run_harness(tmp_path, controller, code, {"x": "(2, 4)"})
    assert rc != 0 and "No nn.Module class found" in out


# ---- shape parsing: the live E2E found the Architect returns "[32, 64, 512]" not "(32, 64, 512)" ----
from sandbox.docker_controller import parse_shape


@pytest.mark.parametrize("raw,expected", [
    ("(32, 64, 512)", (32, 64, 512)), ("[32, 64, 512]", (32, 64, 512)), ("32,64,512", (32, 64, 512)),
    ("32x64x512", (32, 64, 512)), ("torch.Size([2, 3])", (2, 3)), ("(4,)", (4,)), ([2, 3], (2, 3)), ((2, 3), (2, 3)),
])
def test_parse_shape_accepts_llm_formats(raw, expected):
    assert parse_shape(raw) == expected


@pytest.mark.parametrize("bad", ["", "()", "abc", "(0, 4)", "(100000, 100000)", None])
def test_parse_shape_rejects_nonsense(bad):
    with pytest.raises(ValueError):
        parse_shape(bad)


def test_harness_builds_valid_call_from_bracket_shapes(controller):
    h = controller._generate_execution_harness(MODEL, {"x": "[2, 4]"})
    assert "torch.randn(2, 4).cuda()" in h
    compile(h, "main_run.py", "exec")


def test_harness_rejects_bad_input_name(controller):
    with pytest.raises(ValueError):
        controller._generate_execution_harness(MODEL, {"x; import os": "(2, 4)"})


def test_run_sandbox_reports_invalid_spec_instead_of_crashing(controller):
    ok, res = controller.run_sandbox_execution(MODEL, {"x": "garbage"})     # fails before touching Docker
    assert ok is False and "INVALID SANDBOX INPUT" in res["logs"]


# ---- dtype-aware inputs + constructor contract (live E2E: masked_fill got a float mask; __init__ needed args) ----
from sandbox.docker_controller import input_dtype, input_expression


@pytest.mark.parametrize("name,spec,kind", [
    ("x", "[4, 8]", "float"), ("key_padding_mask", "[4, 8]", "bool"), ("attn_mask", "(4,8)", "bool"),
    ("input_ids", "[4, 8]", "int"), ("tokens", "[4,8]", "int"), ("x", "[4, 8] int64", "int"),
    ("weird", "[4, 8] bool", "bool"), ("mask", "[4, 8] float32", "float"),
])
def test_input_dtype_rules(name, spec, kind):
    assert input_dtype(name, spec) == kind


def test_input_expressions_are_valid_and_dtype_correct():
    assert input_expression("m", "[2, 3] bool") == "torch.zeros(2, 3, dtype=torch.bool).cuda()"
    assert input_expression("ids", "[2, 3] int64") == "torch.randint(0, 8, (2, 3,)).cuda()"
    assert input_expression("x", "[2, 3]") == "torch.randn(2, 3).cuda()"
    assert "(4,)" in input_expression("ids", "[4]")


NEEDS_ARGS = textwrap.dedent("""\
    import torch.nn as nn
    class M(nn.Module):
        def __init__(self, d_model, h, dropout=0.1):
            super().__init__()
            self.fc = nn.Linear(d_model, d_model)
        def forward(self, x):
            return self.fc(x)
""")


def test_required_init_args_give_an_actionable_error(tmp_path, controller):
    pytest.importorskip("torch")
    rc, out = _run_harness(tmp_path, controller, NEEDS_ARGS, {"x": "(2, 4)"})
    assert rc != 0 and "d_model" in out and "Give every __init__ parameter a default" in out


def test_defaults_everywhere_runs_and_mask_is_boolean(tmp_path, controller):
    pytest.importorskip("torch")
    code = textwrap.dedent("""\
        import torch, torch.nn as nn
        class M(nn.Module):
            def __init__(self, d=4):
                super().__init__()
                self.fc = nn.Linear(d, d)
            def forward(self, x, mask):
                assert mask.dtype == torch.bool
                return self.fc(x).masked_fill(mask, float('-inf'))
    """)
    rc, out = _run_harness(tmp_path, controller, code, {"x": "[2, 4]", "mask": "[2, 4]"})
    assert rc == 0 and "forward pass success" in out, out

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
    h = controller._generate_execution_harness(MODEL, {"x": "(2, 4)", "mask": "(2, 4)"})
    compile(h, "main_run.py", "exec")
    assert "mask=torch.randn(2, 4).cuda()" in h


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

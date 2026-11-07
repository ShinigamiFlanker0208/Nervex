"""Static analysis must catch errors that ast.parse alone misses (live E2E regression)."""
import pytest
from agents.critic import CriticAgent


@pytest.fixture
def critic():
    return CriticAgent.__new__(CriticAgent)      # skip __init__: it builds an LLM client


@pytest.mark.parametrize("code", [
    "def f(self, key, key):\n    return key\n",            # duplicate argument: parses, does not compile
    "return 1\n",                                             # return outside function
    "def f():\n    nonlocal x\n",                             # nonlocal at module level scope
    "x = (\n",                                                # plain syntax error
    "x = 1\x00\n",                                            # null byte -> ValueError
])
def test_compile_level_errors_are_rejected(critic, code):
    ok, msg = critic.run_static_analysis(code)
    assert ok is False and msg.startswith("Syntax Error")


def test_valid_code_and_whitelist_still_work(critic):
    assert critic.run_static_analysis("import torch\nx = torch.ones(1)\n")[0] is True
    ok, msg = critic.run_static_analysis("import os\n")
    assert ok is False and "Unauthorized import" in msg

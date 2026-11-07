"""AST theory mapper: deterministic comment->node binding. Why: the Explainer's grounding depends on it."""
import textwrap
from explainer.ast_parser import CodeTheoryMapper

SRC = textwrap.dedent('''\
    import torch.nn as nn

    class Block(nn.Module):
        """Implements: <Node_ID: Eq_1>"""
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def forward(self, x):
            # Implements: <Node_ID: Eq_3>
            # [B, S, D] -> [B, S, D]
            h = self.fc(x)
            y = h + x  # Implements: <Node_ID: residual_01>
            z = y * 2
            return z
''')


def mapper():
    m = CodeTheoryMapper(SRC)
    m.parse_and_map()
    return m


def test_comment_above_binds_node_and_shape():
    m = mapper().mappings
    assign = m[12]
    assert assign.ast_type == "Assign"
    assert assign.mapped_graph_node_id == "Eq_3"
    assert assign.inferred_tensor_shape == "[B, S, D] -> [B, S, D]"


def test_inline_comment_binds():
    assert mapper().mappings[13].mapped_graph_node_id == "residual_01"


def test_untagged_assign_not_registered():
    assert 14 not in mapper().mappings


def test_docstring_tag_on_class():
    assert mapper().mappings[3].mapped_graph_node_id == "Eq_1"


def test_range_query_overlap_and_dedupe():
    ctx = mapper().get_context_for_line_range(12, 13)
    assert ctx["node_ids"] == ["Eq_1", "Eq_3", "residual_01"]
    assert ctx["tensor_shapes"] == ["[B, S, D] -> [B, S, D]"]
    assert "h = self.fc(x)" in ctx["selected_code"]


def test_reversed_range_is_normalised():
    assert mapper().get_context_for_line_range(13, 12)["node_ids"] == ["Eq_1", "Eq_3", "residual_01"]


def test_untagged_line_inherits_only_enclosing_class_tag():
    # line 14 (`z = y * 2`) has no tag; the enclosing class (Eq_1) does, forward's header does not.
    # Crucially Eq_3 (a comment INSIDE forward's body, bound to `h`) must NOT leak onto the def.
    ctx = mapper().get_context_for_line_range(14, 14)
    assert ctx["node_ids"] == ["Eq_1"]


def test_class_does_not_steal_inner_method_tag():
    assert mapper().mappings[3].mapped_graph_node_id == "Eq_1"
    assert mapper().mappings[9].mapped_graph_node_id is None


def test_syntax_error_propagates():
    import pytest
    with pytest.raises(SyntaxError):
        CodeTheoryMapper("def (:").parse_and_map()

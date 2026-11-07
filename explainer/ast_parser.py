"""
Abstract Syntax Tree (AST) Parser & Theory Mapper

Parses AI-generated Python backend code into an AST structure, mapping specific classes, methods,
and arithmetic tensor lines directly back to original research paper equations and ontology Node IDs
using embedded metadata comments (`# Implements: <Node_ID: Eq_3>`).
"""

import ast
import re
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

# TODO: Import Graph node references and project config
# from db.graph_db import Neo4jGraphManager


@dataclass
class ASTNodeMapping:
    """
    Data structure representing a parsed code segment linked to its foundational math node.
    """
    line_start: int
    line_end: int
    code_snippet: str
    ast_type: str                   # "FunctionDef", "ClassDef", "Call", "Assign"
    mapped_graph_node_id: Optional[str] = None
    inferred_tensor_shape: Optional[str] = None


class CodeTheoryMapper(ast.NodeVisitor):
    """
    AST Visitor class designed to traverse Python syntactical nodes and bind comments to code blocks.
    """

    def __init__(self, source_code: str):
        """
        Initializes parser state with raw source script and line tokenization tables.
        # TODO: Store source_code, split into line arrays for fast coordinate index lookups.
        # TODO: Initialize dictionary table: `self.mappings: Dict[int, ASTNodeMapping] = {}`.
        """
        self.source_code = source_code
        self.lines = source_code.split("\n")
        self.mappings: Dict[int, ASTNodeMapping] = {}
        # `# Implements: <Node_ID: Eq_3>` (also accepts `# Implements: Eq_3`)
        self.implements_re = re.compile(r"#\s*Implements:\s*<?\s*(?:Node_ID:\s*)?([A-Za-z0-9_\-.]+)\s*>?")
        # `# [B, S, D] -> [B, S, D]`
        self.shape_re = re.compile(r"#\s*(\[[^\]]*\]\s*->\s*.+?)\s*$")

    def _tags_for(self, start_line: int, end_line: int, docstring: Optional[str] = None) -> Tuple[Optional[str], Optional[str]]:
        """
        Find (node_id, shape) for a statement: comments on its own lines (inline) or in the
        contiguous comment block directly above it. Pure text scan, no LLM.
        """
        candidates: List[str] = []
        # contiguous comment lines immediately above (1-indexed -> list index start_line-2)
        i = start_line - 2
        while i >= 0 and self.lines[i].strip().startswith("#"):
            candidates.append(self.lines[i])
            i -= 1
        # inline comments on the statement's own lines
        candidates.extend(self.lines[start_line - 1:end_line])
        if docstring:
            candidates.extend("# " + ln.strip() for ln in docstring.splitlines())

        node_id = shape = None
        for text in candidates:
            if node_id is None and (m := self.implements_re.search(text)):
                node_id = m.group(1)
            if shape is None and (m := self.shape_re.search(text)):
                shape = m.group(1)
        return node_id, shape

    def _register(self, node: ast.AST, ast_type: str, docstring: Optional[str] = None, always: bool = False) -> None:
        start = node.lineno
        end = getattr(node, "end_lineno", start) or start
        scan_end = end
        body = getattr(node, "body", None)
        if ast_type in ("FunctionDef", "ClassDef") and body:
            # Only the HEADER may carry tags. Scanning the whole body would let a class steal
            # the `# Implements:` of a method inside it.
            limit = max(start, body[0].lineno - 1)
            scan_end = limit
            for ln in range(start, limit + 1):          # header ends at the line whose code ends with ':'
                if self.lines[ln - 1].split("#")[0].rstrip().endswith(":"):
                    scan_end = ln
                    break
        node_id, shape = self._tags_for(start, scan_end, docstring)
        if node_id is None and shape is None and not always:
            return
        self.mappings[start] = ASTNodeMapping(
            line_start=start,
            line_end=end,
            code_snippet="\n".join(self.lines[start - 1:end]),
            ast_type=ast_type,
            mapped_graph_node_id=node_id,
            inferred_tensor_shape=shape,
        )

    def parse_and_map(self) -> Dict[int, Any]:
        """
        Executes AST generation and initiates traversal over syntax tree nodes.
        
        Returns:
            Dict[int, Any]: Mapping of 1-indexed line numbers to enriched ASTNodeMapping properties.
            
        # TODO: Run `tree = ast.parse(self.source_code)`.
        # TODO: Call `self.visit(tree)` to populate self.mappings.
        # TODO: Return pre-computed lookup dictionary for O(1) frontend tooltip rendering.
        """
        tree = ast.parse(self.source_code)  # SyntaxError propagates: caller decides how to report
        self.mappings = {}
        self.visit(tree)
        return self.mappings

    def visit_Assign(self, node: ast.Assign) -> Any:
        """
        Visits assignment lines (e.g., `out = self.gelu(self.fc(x))`) and inspects preceeding comments for Math IDs.
        # TODO: Get node.lineno and node.end_lineno.
        # TODO: Inspect line directly above assignment for `# Implements:` markers or inline comments.
        # TODO: Register discovered links in self.mappings.
        """
        self._register(node, "Assign")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """
        Visits function and method definitions (e.g., `def forward(self, x):`) to map whole execution block boundaries.
        # TODO: Extract docstrings using `ast.get_docstring(node)` and parse out theoretical explanations.
        """
        self._register(node, "FunctionDef", ast.get_docstring(node), always=True)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._register(node, "ClassDef", ast.get_docstring(node), always=True)
        self.generic_visit(node)

    def get_context_for_line_range(self, start_line: int, end_line: int) -> Dict[str, Any]:
        """
        Queries pre-computed mappings to locate all Knowledge Graph entity IDs associated with user highlighted code.
        
        Args:
            start_line (int): Start of user highlight selection in frontend code editor.
            end_line (int): End of highlight selection.
            
        Returns:
            Dict[str, Any]: Compiled list of Node IDs, formula names, and tensor shapes active within selection.
        """
        if start_line > end_line:
            start_line, end_line = end_line, start_line
        if not self.mappings:
            self.parse_and_map()

        hits = [
            m for m in self.mappings.values()
            if m.line_start <= end_line and m.line_end >= start_line   # interval overlap
        ]
        hits.sort(key=lambda m: (m.line_start, -m.line_end))
        node_ids = list(dict.fromkeys(m.mapped_graph_node_id for m in hits if m.mapped_graph_node_id))
        shapes = list(dict.fromkeys(m.inferred_tensor_shape for m in hits if m.inferred_tensor_shape))
        return {
            "node_ids": node_ids,
            "tensor_shapes": shapes,
            "segments": [asdict(m) for m in hits],
            "selected_code": "\n".join(self.lines[max(start_line - 1, 0):end_line]),
        }

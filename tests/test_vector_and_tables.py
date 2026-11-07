"""add_math_explanations (fake collection) and markdown-table extraction (pure)."""
from db.vector_db import VectorStoreManager
from ingestion.pdf_parser import PDFIngestionEngine


class Col:
    def __init__(self): self.kw = None
    def upsert(self, **kw): self.kw = kw


def mgr():
    m = VectorStoreManager.__new__(VectorStoreManager); m.collection = Col(); return m


def test_formulas_deduped_flat_metadata_and_stable_ids():
    m = mgr()
    vault = [{"token_id": "<MATH_BLOCK_0000>", "original_latex": "a^2+b^2", "block_type": "inline_math", "approx_line_num": 3},
             {"token_id": "<MATH_BLOCK_0001>", "original_latex": "a^2+b^2", "block_type": "inline_math", "approx_line_num": 9},
             {"token_id": "<MATH_BLOCK_0002>", "original_latex": "  ", "block_type": "inline_math", "approx_line_num": 1}]
    m.add_math_explanations(vault)
    kw = m.collection.kw
    assert len(kw["ids"]) == 1 and kw["metadatas"][0]["entity_class"] == "Formula"
    assert all(isinstance(v, (str, int, float, bool)) for v in kw["metadatas"][0].values())
    m2 = mgr(); m2.add_math_explanations(vault)
    assert m2.collection.kw["ids"] == kw["ids"]            # idempotent across runs


def test_empty_vault_is_noop():
    m = mgr(); m.add_math_explanations([]); assert m.collection.kw is None


MD = """intro
| a | b |
|---|:-:|
| 1 | 2 |
| 3 | 4 |

text with a | pipe but no table
| x |
not-a-separator
"""

def test_extract_markdown_tables():
    t = PDFIngestionEngine.extract_markdown_tables(MD)
    assert len(t) == 1 and t[0].splitlines()[0] == "| a | b |" and t[0].count("\n") == 3

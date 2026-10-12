import sys
import os
import unittest
from pathlib import Path

# Add project root to path so imports work cleanly
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from ingestion.math_masker import MathMasker, MathBlock
from config import settings


class TestCheckpoint1(unittest.TestCase):
    def test_01_config_directories(self):
        print("\n--- [Test 1] Verifying Config Directory Creation ---")
        settings.create_directories()
        self.assertTrue(settings.UPLOAD_DIR.exists(), f"{settings.UPLOAD_DIR} should exist")
        self.assertTrue(settings.ARTIFACTS_DIR.exists(), f"{settings.ARTIFACTS_DIR} should exist")
        self.assertTrue(settings.VECTOR_DB_PATH.exists(), f"{settings.VECTOR_DB_PATH} should exist")
        print("✅ Config directories created and verified successfully!")

    def test_02_math_masker_symmetry(self):
        print("\n--- [Test 2] Verifying MathMasker Tokenization & Restoration Symmetry ---")
        masker = MathMasker()
        
        sample_markdown = """# 3. Proposed Transformer Attention Layer

In this paper, we compute standard self-attention using the following formulation:

$$
\\text{Attention}(Q, K, V) = \\text{softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V
$$

Where the dimension parameter is defined as $d_k = 64$ for each attention head. Furthermore, the loss optimization loop solves the system:

\\begin{equation}
\\mathcal{L}_{Total} = \\sum_{i=1}^{N} \\| y_i - \\hat{y}_i \\|^2 + \\lambda \\int_{-\\infty}^{\\infty} f(x) dx
\\end{equation}

This concludes our mathematical formulation of the layer semantics.
"""
        # 1. Test Masking
        masked_text, vault = masker.mask_document(sample_markdown)
        print("Masked text output snippet:")
        for line in masked_text.strip().split("\n"):
            if "MATH_BLOCK" in line or "3. Proposed" in line:
                print(f"   -> {line}")
        
        # Check that tokens were generated
        self.assertTrue(len(vault) >= 3, f"Expected at least 3 math blocks extracted, got {len(vault)}")
        self.assertIn("<MATH_BLOCK_0000>", masked_text)
        self.assertNotIn("\\text{Attention}", masked_text, "Raw LaTeX should be removed from masked markdown")
        self.assertNotIn("\\mathcal{L}_{Total}", masked_text, "Equation environment should be masked")

        # 2. Test Vault Summary Exporter
        summary = masker.get_vault_summary()
        self.assertEqual(len(summary), len(vault), "Summary length must match vault length")
        self.assertTrue(all("token_id" in d and "original_latex" in d for d in summary), "Summary items must be dicts with required keys")
        
        # 3. Test Unmasking Restoration
        restored_text = masker.unmask_document(masked_text)
        self.assertEqual(sample_markdown, restored_text, "Unmasked text must match the exact original input string with 0% corruption!")
        print("✅ MathMasker achieved 100% loss-free symmetry on complex equations and matrices!")


if __name__ == "__main__":
    unittest.main(verbosity=2)

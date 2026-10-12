import sys
import os
import unittest
from pathlib import Path

# Add project root to path so imports work cleanly
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from ingestion.math_masker import MathMasker
from ingestion.semantic_chunker import SemanticChunker


class TestCheckpoint2(unittest.TestCase):
    def test_semantic_chunking(self):
        print("\n--- [Test 1] Verifying Semantic Chunker Header Splitting ---")
        
        # 1. Simulate a large paper with math
        mock_paper = """# 1. Introduction
This is a paper about neural networks. Let's look at the core formulation:
$$
\\mathcal{L} = -\\sum y \\log(\\hat{y})
$$

## 1.1 Background
The background relies on heavily vectorized operations. Here is an inline equation $x = y^2$ that must not be split across chunks.

# 2. Methodology
Our methodology proposes a new attention mask:
\\begin{align}
A = QK^T / \\sqrt{d}
\\end{align}
This is a very long section that goes on and on. """ * 10 # Multiplied to test character splitting!

        # 2. Mask the math first! (Testing pipeline integration)
        masker = MathMasker()
        masked_markdown, vault = masker.mask_document(mock_paper)
        
        # 3. Initialize Chunker
        chunker = SemanticChunker(chunk_size=150, chunk_overlap=20)
        
        # Global document metadata
        global_metadata = {"title": "Mock Neural Paper", "author": "SoulGraph Test Suite"}
        
        # 4. Chunk it!
        chunks = chunker.chunk_markdown(masked_markdown, global_metadata)
        
        print(f"Total Chunks Generated: {len(chunks)}")
        
        # Validations
        self.assertTrue(len(chunks) > 10, "Chunker should have split the long document into many pieces")
        
        # Check first chunk structure
        first_chunk = chunks[0]
        self.assertIn("chunk_id", first_chunk)
        self.assertIn("content", first_chunk)
        self.assertIn("section_header", first_chunk)
        self.assertIn("math_tokens_present", first_chunk)
        self.assertIn("metadata", first_chunk)
        
        # Verify metadata inheritance and header capture
        self.assertEqual(first_chunk["metadata"]["title"], "Mock Neural Paper")
        
        # Check that math tokens were correctly recorded!
        math_tokens_found = 0
        for c in chunks:
            if len(c["math_tokens_present"]) > 0:
                math_tokens_found += len(c["math_tokens_present"])
                
        self.assertTrue(math_tokens_found > 0, "Chunker failed to identify math tokens inside the chunks!")
        print("✅ Semantic Chunker successfully navigated markdown headers, character limits, and token extraction!")

if __name__ == "__main__":
    unittest.main(verbosity=2)

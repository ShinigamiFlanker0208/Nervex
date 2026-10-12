import sys
import os
from pathlib import Path

# Add project root to path so imports work cleanly
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from ingestion.pdf_parser import PDFIngestionEngine
from ingestion.math_masker import MathMasker
from ingestion.semantic_chunker import SemanticChunker

def run_end_to_end_test():
    print("\n" + "="*50)
    print("🚀 SOULGRAPH PIPELINE END-TO-END TEST 🚀")
    print("="*50)
    
    pdf_path = Path('attention.pdf')
    if not pdf_path.exists():
        print("❌ Error: test PDF not found at data/attention.pdf")
        return

    # --- 1. OCR Ingestion ---
    print("\n[STAGE 1]: Initializing Deep Learning OCR Engine...")
    print("⚠️  NOTE: If this is the first run, Marker will download Gigabytes of HuggingFace models. This may take a few minutes!")
    try:
        ocr_engine = PDFIngestionEngine(use_cuda=False) # CPU fallback for safe testing
        ocr_results = ocr_engine.parse_pdf(pdf_path)
        raw_markdown = ocr_results["raw_markdown"]
        print(f"✅ OCR Complete! Extracted {len(raw_markdown)} characters of markdown.")
    except Exception as e:
        print(f"❌ OCR Engine Failed: {e}")
        return

    # --- 2. Math Masking ---
    print("\n[STAGE 2]: Masking Mathematical Equations...")
    try:
        masker = MathMasker()
        masked_markdown, vault = masker.mask_document(raw_markdown)
        print(f"✅ Masking Complete! Vaulted {len(vault)} mathematical equations/matrices.")
    except Exception as e:
        print(f"❌ Math Masker Failed: {e}")
        return

    # --- 3. Semantic Chunking ---
    print("\n[STAGE 3]: Splitting via Semantic Chunker...")
    try:
        chunker = SemanticChunker(chunk_size=1000, chunk_overlap=150)
        metadata = {"source": "Attention Is All You Need", "authors": "Vaswani et al."}
        chunks = chunker.chunk_markdown(masked_markdown, metadata)
        print(f"✅ Chunking Complete! Generated {len(chunks)} contextual chunks.")
        
        # Display sample chunk
        print("\n--- SAMPLE CHUNK (Index 5) ---")
        sample = chunks[5] if len(chunks) > 5 else chunks[0]
        print(f"Header Context: {sample.get('section_header', {})}")
        print(f"Math Tokens Found: {sample.get('math_tokens_present', [])}")
        print(f"Preview: {sample['content'][:200]}...")
        
    except Exception as e:
        print(f"❌ Semantic Chunker Failed: {e}")
        return

    print("\n" + "="*50)
    print("🏆 FULL INGESTION PIPELINE VERIFIED SUCCESSFULLY 🏆")
    print("="*50)

if __name__ == "__main__":
    run_end_to_end_test()

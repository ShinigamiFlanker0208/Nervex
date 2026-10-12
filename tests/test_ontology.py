from ingestion.ontology_extractor import OntologyExtractor

def run_test():
    print("🚀 Firing up NVIDIA Nemotron Ontology Extractor...")
    extractor = OntologyExtractor()
    
    chunk_text = "The Transformer architecture relies entirely on the Multi-Head Attention layer. This layer implements the scaled dot-product attention equation: <MATH_BLOCK_0>."
    math_vault = {
        "<MATH_BLOCK_0>": "Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V"
    }
    
    print(f"\n[STAGE 1]: Processing Chunk:\n{chunk_text}")
    print("\nExtracting Graph nodes and edges...")
    schema = extractor.extract_from_chunk(chunk_text, math_vault)
    
    print("\n✅ Extraction successful! Extracted Schema:")
    print(schema.model_dump_json(indent=2))

if __name__ == "__main__":
    run_test()

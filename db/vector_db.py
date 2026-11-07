"""
Vector Database Engine (ChromaDB / Milvus Integration)

Manages local embeddings for document chunks and natural language explanations of LaTeX formulas
to enable hybrid semantic vector searches that complement graph traversals.
"""

import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

# TODO: Import ChromaDB client modules and sentence-transformers embedding functions
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer

# TODO: Import project configurations
from config import settings


class VectorStoreManager:
    """
    Manages semantic vector collection lifecycle, embedding model inference, and hybrid vector indexing.
    """



    def __init__(self, persistence_dir: Path, collection_name: str, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes a local persistent ChromaDB/Milvus client and loads sentence embedding weights.
        # TODO: Initialize ChromaDB persistent client at persistence_dir.
        # TODO: Configure sentence-transformers embedding function using model_name.
        # TODO: Get or create target collection (self.client.get_or_create_collection).
        """
        self.persistence_dir =settings.VECTOR_DB_PATH
        self.collection_name =settings.COLLECTION_NAME
        self.model_name = settings.EMBEDDING_MODEL_NAME
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.model_name)
        self.client =chromadb.PersistentClient(path=str(persistence_dir))
        self.collection =self.client.get_or_create_collection(name=self.collection_name,embedding_function=ef)
        print("[VectorStoreManager] Mock initialized for testing.")

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Embeds and writes structured markdown chunks into the vector collection.
        
        Args:
            chunks (List[Dict[str, Any]]): Chunk dictionaries produced by SemanticChunker.
            
        # TODO: Extract text payloads ("content"), unique identities ("chunk_id"), and structural metadata.
        # TODO: Sanitize metadata dictionaries (ChromaDB requires flat string/int variables, not nested arrays).
        # TODO: Execute collection.upsert(ids=ids, documents=texts, metadatas=clean_metadatas).
        """
        ids=[c["chunk_id"] for c in chunks]
        texts=[c["content"] for c in chunks]
        clean_metadata=[c["metadata"] for c in chunks]
        self.collection.upsert(ids=ids, documents=texts, metadatas=clean_metadata)

    def add_math_explanations(self, math_vault_summary: List[Dict[str, Any]]) -> None:
        """
        Specialized embedding routine that stores generated natural language explanations of LaTeX math blocks.
        
        Args:
            math_vault_summary (List[Dict[str, Any]]): Math block definitions containing LLM descriptions.
            
        # TODO: Format custom documents embedding both original LaTeX strings and English explanations.
        # TODO: Upsert into collection with tag `entity_class='Formula'`.
        """
        seen, ids, docs, metas = set(), [], [], []
        for block in math_vault_summary:
            latex = (block.get("original_latex") or "").strip()
            if not latex or latex in seen:           # dedupe: the same formula recurs across the paper
                continue
            seen.add(latex)
            kind = block.get("block_type", "unknown")
            explanation = (block.get("explanation") or "").strip()   # optional English gloss, if one exists
            docs.append(f"Formula ({kind}): {latex}" + (f"\nMeaning: {explanation}" if explanation else ""))
            # content-addressed id -> re-ingesting the same paper upserts instead of duplicating
            ids.append("formula_" + hashlib.sha1(latex.encode("utf-8")).hexdigest()[:16])
            metas.append({
                "entity_class": "Formula",           # Chroma metadata must be flat str/int/float/bool
                "block_type": str(kind),
                "token_id": str(block.get("token_id", "")),
                "approx_line_num": int(block.get("approx_line_num", 0) or 0),
            })
        if ids:
            self.collection.upsert(ids=ids, documents=docs, metadatas=metas)

    def similarity_search(self, query: str, n_results: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Executes semantic vector similarity searches against embedded document knowledge.
        
        Args:
            query (str): Natural language research query or code error message.
            n_results (int): Max number of candidate passages to retrieve.
            filter_metadata (Optional[Dict[str, Any]]): Exact match filters (e.g., {"section_header": "2. Method"}).
            
        Returns:
            List[Dict[str, Any]]: Ranked retrieval items containing distance scores, matched text, and source IDs.
            
        # TODO: Execute self.collection.query(query_texts=[query], n_results=n_results, where=filter_metadata).
        # TODO: Parse Chroma search output into clean, iterable list of dictionaries with distance metrics.
        """
        results=self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_metadata
        )
        documents = results.get('documents', [[]])[0]
        distances = results.get('distances', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]

        # displaying the results
        output=[]

        for doc , dis, meta in zip(documents , distances , metadatas):
            output.append({
                "content":doc,
                "distance":dis,
                "metadata":meta
            })
        return output


    def paper_passages(self, paper_id: str, limit: int = 400) -> List[Dict[str, Any]]:
        """Every stored chunk of one paper, in reading order (chunk ids are `<paper_id>_chunk_0000`, zero-padded,
        so sorting by id IS document order). Used by the UI to rebuild the paper's text."""
        got = self.collection.get(where={"paper_id": paper_id}, include=["documents", "metadatas"], limit=limit)
        rows = [{"id": i, "text": d or "", "meta": m or {}} for i, d, m in zip(got.get("ids", []), got.get("documents", []), got.get("metadatas", []))]
        rows = [r for r in rows if "_chunk_" in r["id"]]          # skip math-explanation docs
        return sorted(rows, key=lambda r: r["id"])

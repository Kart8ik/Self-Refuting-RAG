import os
import faiss
import numpy as np
import json
from typing import List, Dict
from sr_rag.models import RetrievedPassage
from sr_rag.retrieval.embedding_model import EmbeddingModel

class VectorIndex:
    def __init__(self, embedding_model: EmbeddingModel = None):
        if embedding_model is None:
            self.embedding_model = EmbeddingModel()
        else:
            self.embedding_model = embedding_model
        
        self.d = 384
        self.M = 16

        # Reduce native thread contention; improves stability on macOS arm64.
        faiss_threads = int(os.environ.get("FAISS_THREADS", "1"))
        try:
            faiss.omp_set_num_threads(max(1, faiss_threads))
        except Exception:
            pass

        # Default to FLAT for stability; set SR_RAG_FAISS_INDEX=hnsw to re-enable HNSW.
        index_mode = os.environ.get("SR_RAG_FAISS_INDEX", "flat").strip().lower()
        if index_mode == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.d, self.M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.hnsw.efSearch = 50
        else:
            self.index = faiss.IndexFlatIP(self.d)

        self.metadata: List[Dict] = []
        
    def build(self, documents: List[str], metadata: List[Dict]) -> None:
        assert len(documents) == len(metadata), "Documents and metadata length must match"
        if not documents:
            return
            
        embeddings = self.embedding_model.encode(documents)
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        
    def retrieve(self, query_text: str, k: int = 5) -> List[RetrievedPassage]:
        if self.index.ntotal == 0:
            return []
            
        k = min(k, self.index.ntotal)
        query_emb = self.embedding_model.encode([query_text])
        
        # Search index
        scores, indices = self.index.search(query_emb, k)
        
        passages = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            passage = RetrievedPassage(
                doc_id=meta.get("doc_id", f"doc_{idx}"),
                source_title=meta.get("source_title", "Unknown"),
                chunk_index=meta.get("chunk_index", 0),
                text=meta.get("text", ""),
                similarity_score=float(score)
            )
            passages.append(passage)
            
        return passages
        
    def save(self, path: str) -> None:
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.json", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)
            
    def load(self, path: str) -> None:
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

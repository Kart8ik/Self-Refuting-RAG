import numpy as np
import re
from typing import List, Dict, Any
from sr_rag.retrieval.embedding_model import EmbeddingModel
from sr_rag.config import load_config

class ClaimValidator:
    def __init__(self, config=None, embedder: EmbeddingModel = None):
        if config is None:
            self.config = load_config()
        else:
            self.config = config
        
        self.embedder = embedder if embedder is not None else EmbeddingModel()
        self.dedup_threshold = getattr(self.config.validation, "dedup_similarity_threshold", 0.92)
        self.min_words = getattr(self.config.validation, "min_claim_words", 8)

    def _has_named_entity(self, text: str) -> bool:
        words = text.split()
        if len(words) <= 1:
            return False
            
        for w in words[1:]:
            w_clean = re.sub(r'[^\w\s]', '', w)
            if w_clean and w_clean[0].isupper():
                return True
                
        if re.search(r'\d+', text):
            return True
            
        return False

    def validate(self, raw_claims: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
        stats = {"original": len(raw_claims), "rejected_vague": 0, "rejected_no_entity": 0, "rejected_duplicate": 0}
        
        filtered_claims = []
        for rc in raw_claims:
            text = rc.get("claim_text", "")
            words = text.split()
            if len(words) < self.min_words:
                stats["rejected_vague"] += 1
                continue
            if not self._has_named_entity(text):
                stats["rejected_no_entity"] += 1
                continue
            filtered_claims.append(rc)
            
        if not filtered_claims:
            return [], stats
            
        texts = [c["claim_text"] for c in filtered_claims]
        embeddings = self.embedder.encode(texts)
        
        sim_matrix = np.dot(embeddings, embeddings.T)
        
        keep_indices = set(range(len(filtered_claims)))
        
        for i in range(len(filtered_claims)):
            if i not in keep_indices:
                continue
            for j in range(i + 1, len(filtered_claims)):
                if j not in keep_indices:
                    continue
                if sim_matrix[i, j] > self.dedup_threshold:
                    len_i = len(filtered_claims[i]["claim_text"])
                    len_j = len(filtered_claims[j]["claim_text"])
                    if len_i >= len_j:
                        keep_indices.remove(j)
                    else:
                        keep_indices.remove(i)
                        break
                        
        stats["rejected_duplicate"] = len(filtered_claims) - len(keep_indices)
        final_claims = [filtered_claims[i] for i in sorted(list(keep_indices))]
        
        return final_claims, stats

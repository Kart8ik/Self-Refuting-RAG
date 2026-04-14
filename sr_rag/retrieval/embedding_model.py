import os
import numpy as np
from typing import List, Union

class EmbeddingModel:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingModel, cls).__new__(cls)
        return cls._instance

    def _load_model(self):
        if self._model is None:
            # Lazy import to avoid loading the model if not used
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
            from sentence_transformers import SentenceTransformer
            try:
                import torch
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            self._model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        self._load_model()
        if isinstance(texts, str):
            texts = [texts]
        
        # encode returns numpy array natively if convert_to_numpy=True (default)
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=16,
        )
        
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1e-10, norms)
        return embeddings / norms

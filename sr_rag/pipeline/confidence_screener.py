import random
from typing import List, Tuple
from sr_rag.models import Claim
from sr_rag.config import load_config

class ConfidenceScreener:
    def __init__(self, config=None):
        if config is None:
            self.config = load_config()
        else:
            self.config = config
            
        self.llm_threshold = getattr(self.config.screening, "llm_confidence_threshold", 0.85)
        self.faiss_threshold = getattr(self.config.screening, "faiss_similarity_threshold", 0.65)
        self.spot_check_rate = getattr(self.config.screening, "spot_check_rate", 0.10)
        self.seed = getattr(self.config.evaluation, "seed", 42)
        
        self.rng = random.Random(self.seed)
        
    def screen(self, claims: List[Claim]) -> Tuple[List[Claim], List[Claim]]:
        refuter_queue = []
        bypass_queue = []
        
        high_conf_claims = []
        
        for c in claims:
            is_low_conf = (c.llm_confidence < self.llm_threshold) or (c.max_passage_similarity < self.faiss_threshold)
            
            if is_low_conf:
                c.routing = "LOW_CONF"
                refuter_queue.append(c)
            else:
                c.routing = "HIGH_CONF"
                high_conf_claims.append(c)
                
        # Spot check HIGH_CONF claims
        for c in high_conf_claims:
            if self.rng.random() < self.spot_check_rate:
                c.spot_check = True
                refuter_queue.append(c)
            else:
                c.spot_check = False
                bypass_queue.append(c)
                
        return refuter_queue, bypass_queue

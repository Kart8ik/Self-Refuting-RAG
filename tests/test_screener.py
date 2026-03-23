import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sr_rag.pipeline.confidence_screener import ConfidenceScreener
from sr_rag.models import Claim
from sr_rag.config import load_config

class TestScreener(unittest.TestCase):
    def test_screener(self):
        config = load_config("sr_rag/config.yaml")
        config.screening.llm_confidence_threshold = 0.85
        config.screening.faiss_similarity_threshold = 0.65
        config.screening.spot_check_rate = 0.50
        config.evaluation.seed = 42
        
        screener = ConfidenceScreener(config)
        
        claims = [
            Claim("c1", "test", 0.80, 0.90, [], "PENDING"),
            Claim("c2", "test", 0.90, 0.60, [], "PENDING"),
            Claim("c3", "test", 0.50, 0.50, [], "PENDING"),
            Claim("c4", "test", 0.90, 0.90, [], "PENDING"),
            Claim("c5", "test", 0.95, 0.95, [], "PENDING"),
            Claim("c6", "test", 0.95, 0.95, [], "PENDING"),
            Claim("c7", "test", 0.95, 0.95, [], "PENDING")
        ]
        
        refuter_queue, bypass_queue = screener.screen(claims)
        
        low_conf = [c for c in claims if c.claim_id in ["c1", "c2", "c3"]]
        for c in low_conf:
            self.assertEqual(c.routing, "LOW_CONF")
            self.assertIn(c, refuter_queue)
            
        high_conf = [c for c in claims if c.claim_id in ["c4", "c5", "c6", "c7"]]
        for c in high_conf:
            self.assertEqual(c.routing, "HIGH_CONF")
            if c.spot_check:
                self.assertIn(c, refuter_queue)
            else:
                self.assertIn(c, bypass_queue)
                
        self.assertTrue(any(c.spot_check for c in high_conf))
        self.assertTrue(any(not c.spot_check for c in high_conf))
        
        print("Confidence Screener tests passed successfully!")

if __name__ == '__main__':
    unittest.main()

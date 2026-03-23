import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sr_rag.pipeline.evidence_scorer import EvidenceScorer
from sr_rag.models import Claim, RetrievedPassage

class TestScorer(unittest.TestCase):
    def test_scorer(self):
        scorer = EvidenceScorer()
        
        claim = Claim("c1", "The year was 1999.", 0.9, 0.9, ["d1", "d2"], "PENDING")
        
        passages = [
            RetrievedPassage("d1", "Book A", 0, "Text 1", 0.8),
            RetrievedPassage("d2", "Book B", 0, "Text 2", 0.6),
            RetrievedPassage("d3", "Book C", 0, "Text 3", 0.9)
        ]
        
        bundle = scorer.score(claim, passages, None)
        
        self.assertAlmostEqual(bundle.relevance_score, 0.7)
        self.assertAlmostEqual(bundle.count_score, 2/3.0)
        self.assertEqual(bundle.specificity_score, 1.0)
        
        expected_prog_conf = 0.40 * 0.7 + 0.35 * (2/3.0) + 0.25 * 1.0
        self.assertAlmostEqual(bundle.programmatic_confidence, expected_prog_conf)
        
        print(f"Computed confidence: {bundle.programmatic_confidence:.4f}")
        print("Evidence Scorer tests passed successfully!")

if __name__ == '__main__':
    unittest.main()

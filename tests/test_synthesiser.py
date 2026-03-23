import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sr_rag.pipeline.output_synthesiser import OutputSynthesiser
from sr_rag.models import Claim, JudgeVerdict, RetrievedPassage

class TestSynthesiser(unittest.TestCase):
    def test_synthesiser(self):
        synth = OutputSynthesiser()
        
        c1 = Claim("c1", "Claim 1 is good.", 0.9, 0.9, [], "PENDING")
        c2 = Claim("c2", "Claim 2 is conflict.", 0.9, 0.9, [], "PENDING")
        c3 = Claim("c3", "Claim 3 is bad.", 0.9, 0.9, [], "PENDING")
        
        p1 = RetrievedPassage("d1", "T", 0, "Prop 1", 0.9)
        p2 = RetrievedPassage("d2", "T", 0, "Count 2", 0.9)
        
        v1 = JudgeVerdict("c1", "SUPPORTED", 0.9, "", [p1], [])
        v2 = JudgeVerdict("c2", "CONFLICTING", 0.5, "", [p1], [p2])
        v3 = JudgeVerdict("c3", "REFUTED", 0.1, "", [p1], [p2])
        
        out = synth.synthesise("query", [c1, c2, c3], [v1, v2, v3], [False, False, False])
        
        self.assertIn("Claim 1 is good.", out.natural_language_answer)
        self.assertIn("[⚠ conflicting evidence — see table]", out.natural_language_answer)
        self.assertNotIn("Claim 3 is bad.", out.natural_language_answer)
        self.assertIn("Note: one or more claims could not be verified and have been omitted.", out.natural_language_answer)
        
        self.assertIsNotNone(out.claim_table)
        self.assertEqual(len(out.claim_table), 2)
        
        self.assertEqual(out.metadata["total_claims"], 3)
        self.assertEqual(out.metadata["supported"], 1)
        self.assertEqual(out.metadata["conflicting"], 1)
        self.assertEqual(out.metadata["refuted"], 1)
        
        print("Output Synthesiser tests passed successfully!")

if __name__ == '__main__':
    unittest.main()

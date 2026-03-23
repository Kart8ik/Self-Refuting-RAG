import sys
import os
import unittest
import json
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sr_rag.agents.judge import JudgeAgent
from sr_rag.models import Claim, RetrievedPassage, RefuterResult, EvidenceScoreBundle
from sr_rag.config import load_config

class TestJudge(unittest.TestCase):
    @patch('sr_rag.agents.judge.ChatGroq')
    def test_judge(self, mock_chat_groq):
        config = load_config("sr_rag/config.yaml")
        agent = JudgeAgent(config)
        agent.llm = MagicMock()
        
        claim = Claim("c1", "Test claim", 0.9, 0.9, [], "PENDING")
        score_bundle = EvidenceScoreBundle("c1", 0.8, 0.8, 1.0, 0.8)
        passages = [RetrievedPassage("d1", "Title", 0, "Prop Text", 0.9)]
        ref_res = RefuterResult("c1", "NOT_FOUND", [], "query", False)
        
        # Test 1: Supported
        agent.llm.invoke.return_value = MagicMock(content='{"verdict": "SUPPORTED", "confidence": 0.9, "justification": "Good"}')
        verdict1 = agent.judge_claim(claim, score_bundle, passages, ref_res)
        self.assertEqual(verdict1.verdict, "SUPPORTED")
        self.assertAlmostEqual(verdict1.final_confidence, (0.9 + 0.8) / 2)
        
        # Test 2: Conflicting
        agent.llm.invoke.return_value = MagicMock(content='{"verdict": "CONFLICTING", "confidence": 0.4, "justification": "Conflict"}')
        verdict2 = agent.judge_claim(claim, score_bundle, passages, ref_res)
        self.assertEqual(verdict2.verdict, "CONFLICTING")
        
        # Test 3: Bad JSON Retry
        agent.llm.invoke.side_effect = [
            MagicMock(content='Not json'),
            MagicMock(content='{"verdict": "UNVERIFIABLE", "confidence": 0.0, "justification": "Fixed"}')
        ]
        verdict3 = agent.judge_claim(claim, score_bundle, passages, ref_res)
        self.assertEqual(verdict3.verdict, "UNVERIFIABLE")
        self.assertEqual(verdict3.justification, "Fixed")
        
        # Test 4: Overall Confidence
        overall = agent.compute_overall_confidence([verdict1, verdict2, verdict3])
        self.assertAlmostEqual(overall, 0.5)
        
        print("Judge tests passed successfully!")

if __name__ == '__main__':
    unittest.main()

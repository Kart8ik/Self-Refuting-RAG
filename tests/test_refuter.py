import sys
import os
import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sr_rag.agents.refuter import RefuterAgent
from sr_rag.models import Claim, RetrievedPassage
from sr_rag.config import load_config
from sr_rag.retrieval.vector_index import VectorIndex

class TestRefuter(unittest.TestCase):
    @patch('sr_rag.agents.refuter.httpx.AsyncClient')
    def test_refuter(self, mock_httpx_client):
        config = load_config("sr_rag/config.yaml")
        agent = RefuterAgent(config)
        
        mock_index = MagicMock(spec=VectorIndex)
        mock_index.retrieve.return_value = [
            RetrievedPassage("d1", "Test", 0, "Counter evidence text.", 0.9)
        ]
        
        mock_client_instance = AsyncMock()
        mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance
        
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        
        # Test 1: Contested claim without leakage
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"verdict": "CONTESTED", "counter_passages": [{"doc_id": "d1", "passage": "Counter evidence text."}], "query_used": "q"}'}}]
        }
        mock_client_instance.post.return_value = mock_response
        
        claim1 = Claim("c1", "Claim one.", 0.9, 0.9, [], "PENDING")
        
        semaphore = asyncio.Semaphore(1)
        res1 = asyncio.run(agent.challenge(claim1, mock_index, semaphore))
        
        self.assertEqual(res1.verdict, "CONTESTED")
        self.assertFalse(res1.leakage_flag)
        self.assertEqual(len(res1.counter_passages), 1)
        self.assertEqual(res1.counter_passages[0].doc_id, "d1")
        
        # Test 2: Leakage detection
        mock_response.json.return_value = {
            "choices": [{"message": {"content": '{"verdict": "CONTESTED", "counter_passages": [{"doc_id": "d99", "passage": "Hallucinated."}], "query_used": "q"}'}}]
        }
        res2 = asyncio.run(agent.challenge(claim1, mock_index, semaphore))
        
        self.assertTrue(res2.leakage_flag)
        
        # Test 3: Exception fallback in challenge_all
        mock_client_instance.post.side_effect = Exception("API Down")
        results = asyncio.run(agent.challenge_all([claim1], mock_index))
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].verdict, "NOT_FOUND")
        self.assertFalse(results[0].leakage_flag)
        
        print("Refuter tests passed successfully!")

if __name__ == '__main__':
    unittest.main()

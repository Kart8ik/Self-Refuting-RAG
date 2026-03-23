import sys
import os
import unittest
from unittest.mock import patch, MagicMock
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sr_rag.pipeline.claim_validator import ClaimValidator
from sr_rag.pipeline.claim_decomposer import ClaimDecomposer
from sr_rag.models import RetrievedPassage
from sr_rag.config import load_config

class TestDecomposer(unittest.TestCase):
    def test_validator(self):
        config = load_config("sr_rag/config.yaml")
        validator = ClaimValidator(config)
        
        raw_claims = [
            {"claim_text": "France has a capital.", "confidence": 1.0, "supporting_doc_ids": []},
            {"claim_text": "the capital of the country is located in a large beautiful city.", "confidence": 1.0, "supporting_doc_ids": []},
            {"claim_text": "Paris is the capital of France and its largest city by population.", "confidence": 0.9, "supporting_doc_ids": []},
            {"claim_text": "Paris is the capital of France and its largest municipality by population.", "confidence": 0.9, "supporting_doc_ids": []}
        ]
        
        valid_claims, stats = validator.validate(raw_claims)
        
        self.assertEqual(len(valid_claims), 1)
        self.assertEqual(stats["rejected_vague"], 1)
        self.assertEqual(stats["rejected_no_entity"], 1)
        self.assertEqual(stats["rejected_duplicate"], 1)

    @patch('sr_rag.pipeline.claim_decomposer.ChatGroq')
    def test_decomposer(self, mock_chat_groq):
        config = load_config("sr_rag/config.yaml")
        decomposer = ClaimDecomposer(config)
        
        mock_instance = MagicMock()
        decomposer.llm = mock_instance
        
        output_json = json.dumps([
            {"claim_text": "The Eiffel Tower was built in Paris in the year 1889.", "confidence": 0.95, "supporting_doc_ids": ["d1"]}
        ])
        mock_instance.invoke.return_value = MagicMock(content=output_json)
        
        passages = [
            RetrievedPassage(doc_id="d1", source_title="D1", chunk_index=0, text="Eiffel Tower was built in 1889.", similarity_score=0.88),
            RetrievedPassage(doc_id="d2", source_title="D2", chunk_index=0, text="Paris is in France.", similarity_score=0.45)
        ]
        
        claims = decomposer.decompose("run_123", "The Eiffel Tower was built in Paris in the year 1889.", passages)
        
        self.assertEqual(len(claims), 1)
        self.assertEqual(claims[0].claim_id, "c_run_123_000")
        self.assertEqual(claims[0].max_passage_similarity, 0.88)
        self.assertEqual(claims[0].routing, "PENDING")
        
        print("Decomposer and Validator tests passed successfully!")

if __name__ == '__main__':
    unittest.main()

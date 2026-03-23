import sys
import os

# Add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
from unittest.mock import patch, MagicMock
from sr_rag.agents.classifier import ClassifierAgent
from sr_rag.config import load_config

class TestClassifier(unittest.TestCase):
    @patch('sr_rag.agents.classifier.ChatGroq')
    def test_classify(self, mock_chat_groq):
        # We need config loaded in test environment
        config = load_config("sr_rag/config.yaml")

        # Create mock responses for LLM
        mock_instance = MagicMock()
        mock_chat_groq.return_value = mock_instance

        agent = ClassifierAgent(config)
        agent.llm = mock_instance  # ensure using the mock
        
        # Test SKIP
        mock_instance.invoke.return_value = MagicMock(content="SKIP")
        self.assertEqual(agent.classify("Hello there!"), "SKIP")
        
        # Test LITE
        mock_instance.invoke.return_value = MagicMock(content=" LITE.")
        self.assertEqual(agent.classify("What is the capital of France?"), "LITE")
        
        # Test FULL
        mock_instance.invoke.return_value = MagicMock(content="FULL ")
        self.assertEqual(agent.classify("Explain the medical implications of CRISPR."), "FULL")
        
        # Test Default to FULL on invalid
        mock_instance.invoke.return_value = MagicMock(content="IDK MAN")
        self.assertEqual(agent.classify("Some query?"), "FULL")
        
        # Test Default to FULL on Exception
        mock_instance.invoke.side_effect = Exception("API Error")
        self.assertEqual(agent.classify("Test error"), "FULL")
        
        print("Classifier tests passed successfully!")

if __name__ == '__main__':
    unittest.main()

import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sr_rag.agents.proposer import ProposerAgent, AbstentionError
from sr_rag.config import load_config
from sr_rag.models import RetrievedPassage

class TestProposer(unittest.TestCase):
    @patch('sr_rag.agents.proposer.ChatGroq')
    def test_proposer(self, mock_chat_groq):
        config = load_config("sr_rag/config.yaml")
        
        agent = ProposerAgent(config)
        agent.llm = MagicMock()
        
        passages_low = [
            RetrievedPassage(doc_id="d1", source_title="D1", chunk_index=0, text="Bad text.", similarity_score=0.10)
        ]
        
        with self.assertRaises(AbstentionError):
            agent.generate("Query?", passages_low)
            
        passages_high = [
            RetrievedPassage(doc_id="d1", source_title="D1", chunk_index=0, text="Good text.", similarity_score=0.90),
            RetrievedPassage(doc_id="d2", source_title="D2", chunk_index=0, text="More good text.", similarity_score=0.85)
        ]
        
        agent.llm.invoke.return_value = MagicMock(content="Here is the answer.")
        answer = agent.generate("What is good?", passages_high)
        
        self.assertEqual(answer, "Here is the answer.")
        self.assertEqual(agent.llm.invoke.call_args[0][0][1].content, "Question: What is good?\n\nRetrieved Documents:\n[1] Good text.\n[2] More good text.")
        
        print("Proposer tests passed successfully!")

if __name__ == '__main__':
    unittest.main()

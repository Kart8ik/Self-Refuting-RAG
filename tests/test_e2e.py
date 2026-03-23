import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import faiss
from sr_rag.config import load_config
from sr_rag.retrieval.vector_index import VectorIndex
from main import run_query

def test_e2e():
    print("Starting e2e tests...")
    
    config = load_config()
    index = VectorIndex()
    
    docs = [
        "The SR-RAG architecture was proposed by Team 20 for their GenAI course in March 2026.",
        "It uses a multi-agent system consisting of Proposer, Decomposer, Refuter, and Judge.",
        "The Refuter operates using only retrieved documents to challenge low confidence claims."
    ]
    metadata = [
        {"doc_id": "d1", "source_title": "Project Info", "chunk_index": 0, "text": docs[0]},
        {"doc_id": "d2", "source_title": "Project Info", "chunk_index": 0, "text": docs[1]},
        {"doc_id": "d3", "source_title": "Project Info", "chunk_index": 0, "text": docs[2]},
    ]
    
    print("Building index...")
    index.build(docs, metadata)
    
    queries = [
        "What is SR-RAG and who proposed it?",
        "Does the system use retrieved documents for the Refuter?",
        "When was SR-RAG proposed?"
    ]
    
    for i, q in enumerate(queries):
        print(f"\n--- Query {i+1}: {q} ---")
        try:
            output = run_query(q, config, index)
            print(f"Answer: {output.natural_language_answer}")
            print(f"Overall Confidence: {output.overall_confidence:.2f}")
            if output.claim_table:
                print("Claim Table:")
                for c in output.claim_table:
                    print(c)
        except Exception as e:
            print(f"Error executing query: {e}")

if __name__ == "__main__":
    if "GROQ_API_KEY" in os.environ and os.environ["GROQ_API_KEY"] != "dummy_key":
        test_e2e()
    else:
        print("Skipping E2E since GROQ_API_KEY is missing")

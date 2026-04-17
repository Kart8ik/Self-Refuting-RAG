import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import faiss
from sr_rag.config import load_config
from sr_rag.retrieval.vector_index import VectorIndex
from sr_rag.retrieval.dataset_loader import (
    load_text_corpus,
    load_text_corpus_from_multiple_datasets,
)
from main import run_query

def test_e2e():
    print("Starting e2e tests...")
    
    config = load_config()
    index = VectorIndex()
    
    dataset_name = os.environ.get("E2E_DATASET_NAME")
    dataset_names_env = os.environ.get("E2E_DATASET_NAMES", "")
    dataset_names = [d.strip() for d in dataset_names_env.split(",") if d.strip()]
    dataset_file = os.environ.get("E2E_DATA_FILE")
    dataset_split = os.environ.get("E2E_DATASET_SPLIT", "train")
    max_docs = int(os.environ.get("E2E_MAX_DOCS", "300"))
    text_fields_env = os.environ.get("E2E_TEXT_FIELDS", "")
    text_fields = [f.strip() for f in text_fields_env.split(",") if f.strip()] or None
    custom_queries_env = os.environ.get("E2E_QUERIES", "")
    custom_queries = [q.strip() for q in custom_queries_env.split("||") if q.strip()]
    using_dataset_corpus = bool(dataset_names or dataset_name or dataset_file)

    if dataset_names:
        print("Loading corpus from multiple datasets...")
        docs, metadata = load_text_corpus_from_multiple_datasets(
            dataset_names=dataset_names,
            split=dataset_split,
            text_fields=text_fields,
            max_docs=max_docs,
        )
        print(f"Loaded {len(docs)} docs from {len(dataset_names)} datasets")
    elif dataset_name or dataset_file:
        print("Loading corpus from dataset...")
        docs, metadata = load_text_corpus(
            dataset_name=dataset_name,
            data_file=dataset_file,
            split=dataset_split,
            text_fields=text_fields,
            max_docs=max_docs,
        )
        print(f"Loaded {len(docs)} docs into index")
    else:
        docs = [
            "The SR-RAG architecture was proposed by Team 20 for their GenAI course in March 2026.",
            "It uses a multi-agent system consisting of Proposer, Decomposer, Refuter, and Judge.",
            "The Refuter operates using only retrieved documents to challenge low confidence claims.",
        ]
        metadata = [
            {"doc_id": "d1", "source_title": "Project Info", "chunk_index": 0, "text": docs[0]},
            {"doc_id": "d2", "source_title": "Project Info", "chunk_index": 0, "text": docs[1]},
            {"doc_id": "d3", "source_title": "Project Info", "chunk_index": 0, "text": docs[2]},
        ]
    
    print("Building index...")
    index.build(docs, metadata)

    if custom_queries:
        queries = custom_queries
    elif using_dataset_corpus:
        # Keep evaluation questions aligned with the chosen corpus.
        queries = docs[:3] if len(docs) >= 3 else docs
    else:
        queries = [
            "What is SR-RAG and who proposed it?",
            "Does the system use retrieved documents for the Refuter?",
            "When was SR-RAG proposed?",
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

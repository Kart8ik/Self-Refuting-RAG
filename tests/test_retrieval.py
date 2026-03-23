import sys
import os

# Add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sr_rag.retrieval.vector_index import VectorIndex
import tempfile
import faiss

def test_retrieval():
    print("Testing embedding model and vector index...")
    
    docs = [
        "The capital of France is Paris.",
        "Python is a popular programming language.",
        "Machine learning is a subset of AI.",
        "The Eiffel Tower is located in Paris.",
        "FAISS is library for efficient similarity search.",
        "Water boils at 100 degrees Celsius.",
        "Jupiter is the largest planet in our solar system.",
        "Photosynthesis is how plants make food.",
        "Albert Einstein developed the theory of relativity.",
        "The Great Wall of China is very long."
    ]
    
    metadata = [
        {"doc_id": f"d{i}", "source_title": f"Doc {i}", "chunk_index": 0, "text": docs[i]} for i in range(len(docs))
    ]
    
    index = VectorIndex()
    index.build(docs, metadata)
    
    print(f"Index built with {index.index.ntotal} documents.")
    
    query = "What is the capital of France?"
    results = index.retrieve(query, k=3)
    
    print(f"\nQuery: {query}")
    for i, res in enumerate(results):
        print(f"{i+1}. {res.text} (Score: {res.similarity_score:.4f})")
        assert 0.0 <= res.similarity_score <= 1.001, f"Score out of bounds: {res.similarity_score}"
        
    assert "Paris" in results[0].text, "Failed to retrieve correct document"
    print("Correctly retrieved the top document.")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "test_index")
        index.save(save_path)
        
        new_index = VectorIndex()
        new_index.load(save_path)
        
        assert new_index.index.ntotal == len(docs), "Loaded index has wrong number of documents"
        assert len(new_index.metadata) == len(docs), "Loaded metadata has wrong number of documents"
        
        results2 = new_index.retrieve(query, k=3)
        assert results[0].doc_id == results2[0].doc_id, "Loaded index retrieves different document"
        print("Save/load works correctly.")
        
    print("\nAll tests passed successfully!")

if __name__ == "__main__":
    test_retrieval()

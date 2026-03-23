import sys
import os

# Add the project root to the python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import tempfile
from sr_rag.pipeline.pipeline_logger import PipelineLogger

def test_pipeline_logger():
    print("Testing Pipeline Logger...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = os.path.join(tmpdir, "logs", "run_log.jsonl")
        logger = PipelineLogger(log_path)
        
        query = "What is SR-RAG?"
        run_id = logger.start_run(query, route="FULL")
        print(f"Started run: {run_id}")
        
        logger.record(run_id, "retrieval", {"passages": ["doc1", "doc2"]})
        logger.record(run_id, "claims", {"claims": ["claim1"]})
        
        final_output = {"natural_language_answer": "SR-RAG is a QA system."}
        logger.finish_run(run_id, final_output, latency_ms=120)
        
        assert os.path.exists(log_path), "Log file was not created"
        
        with open(log_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        assert len(lines) == 1, "Expected 1 line in log file"
        
        log_entry = json.loads(lines[0])
        assert log_entry["run_id"] == run_id
        assert log_entry["query"] == query
        assert log_entry["route"] == "FULL"
        assert log_entry["retrieval"]["passages"] == ["doc1", "doc2"]
        assert log_entry["latency_ms"] == 120
        assert log_entry["final_output"] == final_output
        
        print("Pipeline Logger tests passed successfully!")

if __name__ == "__main__":
    test_pipeline_logger()

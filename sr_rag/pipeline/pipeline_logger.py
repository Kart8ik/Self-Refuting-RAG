import json
import os
from datetime import datetime
from uuid import uuid4

class PipelineLogger:
    def __init__(self, log_path: str = "logs/run_log.jsonl"):
        self.log_path = log_path
        self._runs = {}

    def start_run(self, query: str, route: str) -> str:
        run_id = f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"
        
        self._runs[run_id] = {
            "run_id": run_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "query": query,
            "route": route
        }
        
        return run_id

    def record(self, run_id: str, stage: str, data: dict) -> None:
        if run_id in self._runs:
            self._runs[run_id][stage] = data

    def finish_run(self, run_id: str, final_output: dict, latency_ms: int) -> None:
        if run_id not in self._runs:
            return
            
        run_data = self._runs.pop(run_id)
        run_data["final_output"] = final_output
        run_data["latency_ms"] = latency_ms
        
        os.makedirs(os.path.dirname(os.path.abspath(self.log_path)), exist_ok=True)
        
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(run_data, default=str) + "\n")

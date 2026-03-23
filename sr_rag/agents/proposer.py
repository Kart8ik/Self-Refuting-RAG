import os
import time
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sr_rag.config import load_config
from sr_rag.models import RetrievedPassage

class AbstentionError(Exception):
    pass

class ProposerAgent:
    def __init__(self, config=None):
        if config is None:
            self.config = load_config()
        else:
            self.config = config
            
        self.llm = ChatGroq(
            model=getattr(self.config.llm, "model", "llama-3.3-70b-versatile"),
            temperature=getattr(self.config.llm, "temperature", 0.1),
            max_tokens=getattr(self.config.llm, "max_tokens", 2048),
            api_key=os.environ.get("GROQ_API_KEY", "dummy_key")
        )
        
        prompt_path = os.path.join(
            getattr(self.config.prompts, "base_path", "prompts/"), 
            getattr(self.config.prompts, "version", "v1"), 
            "proposer_system.txt"
        )
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()
            
    def generate(self, query: str, passages: List[RetrievedPassage]) -> str:
        if passages:
            max_sim = max(p.similarity_score for p in passages)
            threshold = getattr(self.config.retrieval, "abstention_threshold", 0.40)
            if max_sim < threshold:
                raise AbstentionError(f"Max similarity {max_sim:.4f} is below abstention threshold {threshold}.")
        else:
            raise AbstentionError("No retrieved passages provided.")
            
        passages_text = "\n".join([f"[{i+1}] {p.text}" for i, p in enumerate(passages)])
        
        user_content = f"Question: {query}\n\nRetrieved Documents:\n{passages_text}"
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content)
        ]
        
        max_retries = getattr(self.config.refuter, "max_retries", 2)
        backoff = getattr(self.config.refuter, "retry_backoff_seconds", [1, 2])
        
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                return response.content.strip()
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "503" in err_str) and attempt < max_retries:
                    wait_time = backoff[attempt] if attempt < len(backoff) else backoff[-1]
                    time.sleep(wait_time)
                    continue
                raise

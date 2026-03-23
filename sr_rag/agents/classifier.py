import os
import time
from typing import Literal
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sr_rag.config import load_config

class ClassifierAgent:
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
            "classifier_system.txt"
        )
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()
            
    def classify(self, query: str) -> Literal["SKIP", "LITE", "FULL"]:
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Query: {query}")
        ]
        
        # Retry logic for all LLM calls on 429 and 503
        max_retries = getattr(self.config.refuter, "max_retries", 2)
        backoff = getattr(self.config.refuter, "retry_backoff_seconds", [1, 2])
        
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                result = response.content.strip().upper()
                
                # Strip punctuation
                import string
                result = result.translate(str.maketrans('', '', string.punctuation))
                
                if result in {"SKIP", "LITE", "FULL"}:
                    return result
                else:
                    return "FULL"
            except Exception as e:
                # If it's a rate limit or service unavailable, try backoff
                err_str = str(e)
                if ("429" in err_str or "503" in err_str) and attempt < max_retries:
                    wait_time = backoff[attempt] if attempt < len(backoff) else backoff[-1]
                    time.sleep(wait_time)
                    continue
                # For any other failure (or final failure), default to FULL
                return "FULL"
        return "FULL"

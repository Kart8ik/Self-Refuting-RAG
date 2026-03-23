import os
import time
import json
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sr_rag.config import load_config
from sr_rag.models import RetrievedPassage, Claim
from sr_rag.pipeline.claim_validator import ClaimValidator

class ClaimDecomposer:
    def __init__(self, config=None, validator=None):
        if config is None:
            self.config = load_config()
        else:
            self.config = config
            
        self.validator = validator if validator is not None else ClaimValidator(self.config)
        
        self.llm = ChatGroq(
            model=getattr(self.config.llm, "model", "llama-3.3-70b-versatile"),
            temperature=getattr(self.config.llm, "temperature", 0.1),
            max_tokens=getattr(self.config.llm, "max_tokens", 2048),
            api_key=os.environ.get("GROQ_API_KEY", "dummy_key")
        )
        
        prompt_path = os.path.join(
            getattr(self.config.prompts, "base_path", "prompts/"), 
            getattr(self.config.prompts, "version", "v1"), 
            "decomposer_system.txt"
        )
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()
            
    def decompose(self, run_id: str, answer_text: str, passages: List[RetrievedPassage]) -> List[Claim]:
        passages_text = "\n".join([f"[{p.doc_id}] {p.text}" for p in passages])
        user_content = f"Answer text:\n{answer_text}\n\nRetrieved Passages:\n{passages_text}\n\nOutput JSON array only, no other text."
        
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content)
        ]
        
        max_retries = getattr(self.config.refuter, "max_retries", 2)
        backoff = getattr(self.config.refuter, "retry_backoff_seconds", [1, 2])
        raw_claims = None
        
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                content = response.content.strip()
                
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                
                raw_claims = json.loads(content.strip())
                if not isinstance(raw_claims, list):
                    raise ValueError("Output must be a JSON array.")
                break
                
            except json.JSONDecodeError:
                if attempt == 0:
                    messages[-1].content += "\n\nCRITICAL: Output ONLY a JSON array. No preamble or markdown."
                    continue
                else:
                    raw_claims = []
                    break
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "503" in err_str) and attempt < max_retries:
                    wait_time = backoff[attempt] if attempt < len(backoff) else backoff[-1]
                    time.sleep(wait_time)
                    continue
                raw_claims = []
                break
                
        if not raw_claims:
            return []
            
        validated_raws, stats = self.validator.validate(raw_claims)
        
        final_claims = []
        for i, r in enumerate(validated_raws):
            claim_id = f"c_{run_id}_{i:03d}"
            supp_docs = r.get("supporting_doc_ids", [])
            
            max_sim = 0.0
            for doc_id in supp_docs:
                for p in passages:
                    if p.doc_id == doc_id and p.similarity_score > max_sim:
                        max_sim = p.similarity_score
                            
            claim = Claim(
                claim_id=claim_id,
                claim_text=r.get("claim_text", ""),
                llm_confidence=float(r.get("confidence", 0.0)),
                max_passage_similarity=max_sim,
                supporting_doc_ids=supp_docs,
                routing="PENDING",
                spot_check=False
            )
            final_claims.append(claim)
            
        return final_claims

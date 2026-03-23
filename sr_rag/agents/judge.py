import os
import time
import json
from typing import List
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sr_rag.config import load_config
from sr_rag.models import Claim, RetrievedPassage, RefuterResult, EvidenceScoreBundle, JudgeVerdict

class JudgeAgent:
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
            "judge_system.txt"
        )
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt_template = f.read().strip()
            
    def judge_claim(self, claim: Claim, score_bundle: EvidenceScoreBundle, proposer_evidence: List[RetrievedPassage], refuter_result: RefuterResult = None) -> JudgeVerdict:
        prop_text = "\n".join([f"[{p.doc_id}] {p.text}" for p in proposer_evidence])
        
        if refuter_result:
            ref_text = f"Verdict: {refuter_result.verdict}\n"
            if refuter_result.counter_passages:
                ref_text += "Counter-evidence:\n" + "\n".join([f"[{p.doc_id}] {p.text}" for p in refuter_result.counter_passages])
        else:
            ref_text = "None (Bypassed refuter)"
            
        system_content = self.system_prompt_template.replace(
            "{relevance_score}", f"{score_bundle.relevance_score:.4f}").replace(
            "{count_score}", f"{score_bundle.count_score:.4f}").replace(
            "{specificity_score}", f"{score_bundle.specificity_score:.4f}").replace(
            "{programmatic_conf}", f"{score_bundle.programmatic_confidence:.4f}").replace(
            "{claim_text}", claim.claim_text).replace(
            "{proposer_evidence}", prop_text).replace(
            "{refuter_result}", ref_text)
        
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content="Evaluate this claim. Output JSON only, no other text.")
        ]
        
        max_retries = getattr(self.config.refuter, "max_retries", 2)
        backoff = getattr(self.config.refuter, "retry_backoff_seconds", [1, 2])
        
        result_json = None
        
        for attempt in range(max_retries + 1):
            try:
                response = self.llm.invoke(messages)
                content = response.content.strip()
                
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                    
                result_json = json.loads(content.strip())
                if "verdict" not in result_json:
                    raise ValueError("Missing verdict in JSON")
                break
                
            except json.JSONDecodeError:
                if attempt == 0:
                    messages[-1].content += "\n\nYou must output ONLY valid JSON with keys: verdict, confidence, justification. No other text."
                    continue
                else:
                    return JudgeVerdict(
                        claim_id=claim.claim_id,
                        verdict="UNVERIFIABLE",
                        final_confidence=0.0,
                        justification="Judge output could not be parsed",
                        supporting_evidence=proposer_evidence,
                        counter_evidence=refuter_result.counter_passages if refuter_result else []
                    )
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "503" in err_str) and attempt < max_retries:
                    wait_time = backoff[attempt] if attempt < len(backoff) else backoff[-1]
                    time.sleep(wait_time)
                    continue
                return JudgeVerdict(
                    claim_id=claim.claim_id,
                    verdict="UNVERIFIABLE",
                    final_confidence=0.0,
                    justification=f"API error: {str(e)[:100]}",
                    supporting_evidence=proposer_evidence,
                    counter_evidence=refuter_result.counter_passages if refuter_result else []
                )
                
        if not result_json:
            return JudgeVerdict(
                claim_id=claim.claim_id,
                verdict="UNVERIFIABLE",
                final_confidence=0.0,
                justification="Failed to get valid result",
                supporting_evidence=proposer_evidence,
                counter_evidence=refuter_result.counter_passages if refuter_result else []
            )
            
        llm_conf = float(result_json.get("confidence", 0.0))
        final_conf = (llm_conf + score_bundle.programmatic_confidence) / 2.0
        
        return JudgeVerdict(
            claim_id=claim.claim_id,
            verdict=result_json.get("verdict", "UNVERIFIABLE"),
            final_confidence=final_conf,
            justification=result_json.get("justification", ""),
            supporting_evidence=proposer_evidence,
            counter_evidence=refuter_result.counter_passages if refuter_result else []
        )
        
    def compute_overall_confidence(self, verdicts: List[JudgeVerdict]) -> float:
        if not verdicts:
            return 0.0
            
        weights = {
            "SUPPORTED": 1.0,
            "CONFLICTING": 0.5,
            "REFUTED": 0.0,
            "UNVERIFIABLE": 0.0
        }
        
        total_weight = sum(weights.get(v.verdict, 0.0) for v in verdicts)
        return total_weight / len(verdicts)

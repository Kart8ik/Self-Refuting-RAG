import os
import time
import json
import re
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

    def _heuristic_fallback_verdict(self, claim: Claim, score_bundle: EvidenceScoreBundle, refuter_result: RefuterResult = None):
        """Deterministic fallback used when judge LLM calls fail (e.g., 429s)."""
        prog = float(score_bundle.programmatic_confidence)
        has_support = bool(claim.supporting_doc_ids)
        refuter_verdict = refuter_result.verdict if refuter_result else "NOT_FOUND"

        if refuter_verdict == "CONTESTED":
            if prog < 0.40:
                verdict = "REFUTED"
                confidence = max(0.20, min(0.80, 0.60 * (1.0 - prog)))
                justification = "Heuristic fallback: refuter found contesting evidence and support was weak."
            else:
                verdict = "CONFLICTING"
                confidence = max(0.25, min(0.85, 0.50 + 0.30 * prog))
                justification = "Heuristic fallback: both supporting and contesting evidence exist."
        elif refuter_verdict == "INSUFFICIENT":
            if has_support and prog >= 0.55:
                verdict = "SUPPORTED"
                confidence = max(0.30, min(0.90, 0.55 + 0.35 * prog))
                justification = "Heuristic fallback: support exists and refuter could not find strong contradiction."
            else:
                verdict = "UNVERIFIABLE"
                confidence = max(0.10, min(0.45, 0.30 * prog))
                justification = "Heuristic fallback: evidence remained insufficient to verify the claim."
        else:  # NOT_FOUND or no refuter result
            if has_support and prog >= 0.45:
                verdict = "SUPPORTED"
                confidence = max(0.30, min(0.92, 0.50 + 0.40 * prog))
                justification = "Heuristic fallback: supporting evidence exists and no contradiction was found."
            elif has_support and prog >= 0.30:
                verdict = "CONFLICTING"
                confidence = max(0.20, min(0.70, 0.35 + 0.35 * prog))
                justification = "Heuristic fallback: support is partial; confidence is limited."
            else:
                verdict = "UNVERIFIABLE"
                confidence = max(0.10, min(0.40, 0.25 * prog))
                justification = "Heuristic fallback: no sufficiently strong supporting evidence was found."

        return verdict, float(confidence), justification

    def _calibrate_verdict(self, claim: Claim, score_bundle: EvidenceScoreBundle, refuter_result: RefuterResult, llm_verdict: str, llm_conf: float):
        """Apply deterministic guardrails so weak/contested evidence is not over-labeled as SUPPORTED."""
        verdict = (llm_verdict or "UNVERIFIABLE").upper()
        refuter_verdict = refuter_result.verdict if refuter_result else "NOT_FOUND"
        prog = float(score_bundle.programmatic_confidence)
        relevance = float(score_bundle.relevance_score)
        has_support = bool(claim.supporting_doc_ids)

        # If refuter found contesting evidence, SUPPORTED is not allowed.
        if refuter_verdict == "CONTESTED" and verdict == "SUPPORTED":
            if prog < 0.40 or relevance < 0.45:
                return "REFUTED", min(llm_conf, 0.55), "Calibrated: refuter reported contested evidence and support score is weak."
            return "CONFLICTING", min(llm_conf, 0.70), "Calibrated: refuter reported contested evidence, so claim cannot remain fully supported."

        # If evidence was deemed insufficient and support is weak, downgrade from SUPPORTED.
        if refuter_verdict == "INSUFFICIENT" and verdict == "SUPPORTED" and (not has_support or prog < 0.55):
            return "UNVERIFIABLE", min(llm_conf, 0.45), "Calibrated: refuter found evidence insufficient and programmatic support was weak."

        # Strong/absolute robustness claims need stronger retrieval support.
        strong_claim = bool(re.search(r"\b(always|never|robust|robustly|large\s+drop-?off|guarantee|cannot|can't)\b", claim.claim_text, flags=re.IGNORECASE))
        if verdict == "SUPPORTED" and strong_claim and relevance < 0.65:
            return "UNVERIFIABLE", min(llm_conf, 0.50), "Calibrated: strong/absolute claim was not backed by sufficiently strong evidence similarity."

        return verdict, llm_conf, ""
            
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
        
        # Ensure we only save the relevant supporting evidence
        claim_supporting_evidence = [p for p in proposer_evidence if p.doc_id in claim.supporting_doc_ids]
        
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
                    verdict, final_conf, justification = self._heuristic_fallback_verdict(claim, score_bundle, refuter_result)
                    return JudgeVerdict(
                        claim_id=claim.claim_id,
                        verdict=verdict,
                        final_confidence=final_conf,
                        justification=f"Judge output could not be parsed; {justification}",
                        supporting_evidence=claim_supporting_evidence,
                        counter_evidence=refuter_result.counter_passages if refuter_result else []
                    )
            except Exception as e:
                err_str = str(e)
                if ("429" in err_str or "503" in err_str) and attempt < max_retries:
                    wait_time = backoff[attempt] if attempt < len(backoff) else backoff[-1]
                    time.sleep(wait_time)
                    continue
                verdict, final_conf, justification = self._heuristic_fallback_verdict(claim, score_bundle, refuter_result)
                return JudgeVerdict(
                    claim_id=claim.claim_id,
                    verdict=verdict,
                    final_confidence=final_conf,
                    justification=f"API error fallback ({str(e)[:100]}): {justification}",
                    supporting_evidence=claim_supporting_evidence,
                    counter_evidence=refuter_result.counter_passages if refuter_result else []
                )
                
        if not result_json:
            verdict, final_conf, justification = self._heuristic_fallback_verdict(claim, score_bundle, refuter_result)
            return JudgeVerdict(
                claim_id=claim.claim_id,
                verdict=verdict,
                final_confidence=final_conf,
                justification=f"Failed to get valid LLM result; {justification}",
                supporting_evidence=claim_supporting_evidence,
                counter_evidence=refuter_result.counter_passages if refuter_result else []
            )
            
        llm_conf = float(result_json.get("confidence", 0.0))
        llm_verdict = result_json.get("verdict", "UNVERIFIABLE")
        calibrated_verdict, calibrated_llm_conf, calibration_reason = self._calibrate_verdict(
            claim,
            score_bundle,
            refuter_result,
            llm_verdict,
            llm_conf,
        )
        final_conf = (calibrated_llm_conf + score_bundle.programmatic_confidence) / 2.0

        justification = result_json.get("justification", "")
        if calibration_reason:
            if justification:
                justification = f"{justification} {calibration_reason}"
            else:
                justification = calibration_reason
        
        return JudgeVerdict(
            claim_id=claim.claim_id,
            verdict=calibrated_verdict,
            final_confidence=final_conf,
            justification=justification,
            supporting_evidence=claim_supporting_evidence,
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

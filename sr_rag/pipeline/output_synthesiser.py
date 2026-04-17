from itertools import zip_longest
from typing import List
from sr_rag.models import Claim, JudgeVerdict, SystemOutput

class OutputSynthesiser:
    def compute_overall_confidence(self, verdicts: List[JudgeVerdict]) -> float:
        if not verdicts:
            return 0.0
        weights = {"SUPPORTED": 1.0, "CONFLICTING": 0.5, "REFUTED": 0.0, "UNVERIFIABLE": 0.0}
        total_weight = sum(weights.get(v.verdict, 0.0) for v in verdicts)
        return total_weight / len(verdicts)

    def synthesise(self, query: str, claims: List[Claim], verdicts: List[JudgeVerdict], leakage_flags: List[bool] = None) -> SystemOutput:
        if leakage_flags is None:
            leakage_flags = []
            
        supported_count = 0
        refuted_count = 0
        conflicting_count = 0
        unverifiable_count = 0
        
        nl_parts = []
        claim_table = []
        
        has_refuted = False
        has_conflicting = False
        
        for c, v in zip_longest(claims, verdicts):
            if c is None:
                continue

            if v is None:
                unverifiable_count += 1
                nl_parts.append(f"{c.claim_text} [could not be verified from available sources]")
                continue

            if v.verdict == "SUPPORTED":
                supported_count += 1
                nl_parts.append(c.claim_text)
            elif v.verdict == "CONFLICTING":
                conflicting_count += 1
                has_conflicting = True
                nl_parts.append(f"{c.claim_text} [⚠ conflicting evidence — see table]")
            elif v.verdict == "UNVERIFIABLE":
                unverifiable_count += 1
                nl_parts.append(f"{c.claim_text} [could not be verified from available sources]")
            elif v.verdict == "REFUTED":
                refuted_count += 1
                has_refuted = True
                
            is_low_conf = v.final_confidence < 0.85

            if v.verdict in ["CONFLICTING", "REFUTED"] or is_low_conf:
                prop_ev = v.supporting_evidence[0].text if v.supporting_evidence else ""
                count_ev = v.counter_evidence[0].text if v.counter_evidence else ""
                
                claim_table.append({
                    "claim": c.claim_text,
                    "verdict": v.verdict,
                    "confidence": f"{v.final_confidence:.2f}",
                    "supporting_evidence": prop_ev,
                    "counter_evidence": count_ev
                })
                
        nl_answer = " ".join(nl_parts)
        if has_refuted:
            nl_answer += "\n\nNote: one or more claims could not be verified and have been omitted."
            
        metadata = {
            "total_claims": len(claims),
            "supported": supported_count,
            "refuted": refuted_count,
            "conflicting": conflicting_count,
            "unverifiable": unverifiable_count,
            "leakage_flags": any(leakage_flags) if leakage_flags else False
        }
        
        overall_conf = self.compute_overall_confidence(verdicts)
        
        return SystemOutput(
            natural_language_answer=nl_answer.strip(),
            overall_confidence=overall_conf,
            claim_table=claim_table if claim_table else None,
            metadata=metadata
        )

import re
from typing import List
from sr_rag.models import Claim, RetrievedPassage, RefuterResult, EvidenceScoreBundle

class EvidenceScorer:
    WEIGHTS = {"relevance": 0.40, "count": 0.35, "specificity": 0.25}

    def score(self, claim: Claim, proposer_passages: List[RetrievedPassage], refuter_result: RefuterResult = None) -> EvidenceScoreBundle:
        supporting_passages = [p for p in proposer_passages if p.doc_id in claim.supporting_doc_ids]
        
        if supporting_passages:
            relevance_score = sum(p.similarity_score for p in supporting_passages) / len(supporting_passages)
        else:
            relevance_score = 0.0
            
        unique_titles = set(p.source_title for p in supporting_passages)
        count_score = min(len(unique_titles) / 3.0, 1.0)
        
        specificity_score = 0.5
        text = claim.claim_text
        if re.search(r'\d+', text):
            specificity_score = 1.0
        else:
            words = text.split()
            if len(words) > 1:
                for w in words[1:]:
                    w_clean = re.sub(r'[^\w\s]', '', w)
                    if w_clean and w_clean[0].isupper():
                        specificity_score = 1.0
                        break
                        
        prog_conf = (
            self.WEIGHTS["relevance"] * relevance_score +
            self.WEIGHTS["count"] * count_score +
            self.WEIGHTS["specificity"] * specificity_score
        )
        
        return EvidenceScoreBundle(
            claim_id=claim.claim_id,
            relevance_score=relevance_score,
            count_score=count_score,
            specificity_score=specificity_score,
            programmatic_confidence=prog_conf
        )

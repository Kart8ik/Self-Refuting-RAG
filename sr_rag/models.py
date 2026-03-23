from dataclasses import dataclass, field
from typing import List, Optional, Literal
from datetime import datetime

@dataclass
class RetrievedPassage:
    doc_id: str
    source_title: str
    chunk_index: int
    text: str
    similarity_score: float

@dataclass
class Claim:
    claim_id: str
    claim_text: str
    llm_confidence: float
    max_passage_similarity: float
    supporting_doc_ids: List[str]
    routing: Literal["HIGH_CONF", "LOW_CONF"]
    spot_check: bool = False

@dataclass
class RefuterResult:
    claim_id: str
    verdict: Literal["CONTESTED", "INSUFFICIENT", "NOT_FOUND"]
    counter_passages: List[RetrievedPassage]
    query_used: str
    leakage_flag: bool = False

@dataclass
class EvidenceScoreBundle:
    claim_id: str
    relevance_score: float
    count_score: float
    specificity_score: float
    programmatic_confidence: float

@dataclass
class JudgeVerdict:
    claim_id: str
    verdict: Literal["SUPPORTED", "REFUTED", "CONFLICTING", "UNVERIFIABLE"]
    final_confidence: float
    justification: str
    supporting_evidence: List[RetrievedPassage]
    counter_evidence: List[RetrievedPassage]

@dataclass
class SystemOutput:
    natural_language_answer: str
    overall_confidence: float
    claim_table: Optional[List[dict]]
    metadata: dict

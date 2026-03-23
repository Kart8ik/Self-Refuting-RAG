import time
import asyncio
from sr_rag.config import load_config
from sr_rag.pipeline.pipeline_logger import PipelineLogger
from sr_rag.agents.classifier import ClassifierAgent
from sr_rag.retrieval.vector_index import VectorIndex
from sr_rag.agents.proposer import ProposerAgent, AbstentionError
from sr_rag.pipeline.claim_decomposer import ClaimDecomposer
from sr_rag.pipeline.confidence_screener import ConfidenceScreener
from sr_rag.agents.refuter import RefuterAgent
from sr_rag.pipeline.evidence_scorer import EvidenceScorer
from sr_rag.agents.judge import JudgeAgent
from sr_rag.pipeline.output_synthesiser import OutputSynthesiser
from sr_rag.models import SystemOutput, Claim

def get_refuter_result(claim: Claim, refuter_results: list):
    for r in refuter_results:
        if r.claim_id == claim.claim_id:
            return r
    return None

def run_query(query: str, config=None, index: VectorIndex = None) -> SystemOutput:
    if config is None:
        config = load_config()
        
    logger = PipelineLogger(getattr(config.logging, "run_log_path", "logs/run_log.jsonl"))
    run_id = logger.start_run(query, route="pending")
    start_time = time.time()
    
    # 1. Classify
    classifier = ClassifierAgent(config)
    route = classifier.classify(query)
    logger.record(run_id, "route", {"route": route})
    
    if run_id in logger._runs:
        logger._runs[run_id]["route"] = route

    if route == "SKIP":
        proposer = ProposerAgent(config)
        try:
            answer_text = proposer.llm.invoke(query).content
        except Exception:
            answer_text = "Sorry, I cannot answer that right now."
            
        output = SystemOutput(
            natural_language_answer=answer_text,
            overall_confidence=1.0,
            claim_table=None,
            metadata={"total_claims": 0, "supported": 0, "refuted": 0, "conflicting": 0, "unverifiable": 0, "leakage_flags": False}
        )
        latency_ms = int((time.time() - start_time) * 1000)
        logger.finish_run(run_id, output.__dict__, latency_ms)
        return output

    # 2. Retrieve
    k = getattr(config.retrieval, "k", 5)
    passages = index.retrieve(query, k=k)
    max_sim = max((p.similarity_score for p in passages), default=0.0)
    logger.record(run_id, "retrieval", {"passages": [p.__dict__ for p in passages], "max_sim": max_sim})

    # 3. Abstention check
    abs_thresh = getattr(config.retrieval, "abstention_threshold", 0.40)
    if max_sim < abs_thresh:
        output = SystemOutput(
            natural_language_answer="Insufficient evidence in knowledge base to answer reliably.",
            overall_confidence=0.0,
            claim_table=None,
            metadata={"total_claims": 0, "supported": 0, "refuted": 0, "conflicting": 0, "unverifiable": 0, "leakage_flags": False}
        )
        latency_ms = int((time.time() - start_time) * 1000)
        logger.finish_run(run_id, output.__dict__, latency_ms)
        return output

    # 4. Propose
    proposer = ProposerAgent(config)
    answer_text = proposer.generate(query, passages)
    logger.record(run_id, "answer", {"answer_text": answer_text})

    # 5. Decompose + validate
    decomposer = ClaimDecomposer(config)
    claims = decomposer.decompose(run_id, answer_text, passages)
    logger.record(run_id, "claims", {"claims": [c.__dict__ for c in claims]})

    if not claims:
        output = SystemOutput(
            natural_language_answer=answer_text + "\n\n[This answer could not be verified.]",
            overall_confidence=0.0,
            claim_table=None,
            metadata={"total_claims": 0, "supported": 0, "refuted": 0, "conflicting": 0, "unverifiable": 0, "leakage_flags": False}
        )
        latency_ms = int((time.time() - start_time) * 1000)
        logger.finish_run(run_id, output.__dict__, latency_ms)
        return output

    # 6. Screen
    screener = ConfidenceScreener(config)
    refuter_queue, bypass_queue = screener.screen(claims)

    # 7. Refute (async)
    refuter = RefuterAgent(config)
    if route == "LITE":
        refuter_results = []
    else:
        refuter_results = asyncio.run(refuter.challenge_all(refuter_queue, index))
    
    logger.record(run_id, "refuter_results", {"results": [r.__dict__ for r in refuter_results]})

    # 8. Score evidence
    evidence_scorer = EvidenceScorer()
    all_score_bundles = {c.claim_id: evidence_scorer.score(c, passages, get_refuter_result(c, refuter_results))
                        for c in claims}

    # 9. Judge
    judge = JudgeAgent(config)
    all_verdicts = [judge.judge_claim(c, all_score_bundles[c.claim_id], passages, get_refuter_result(c, refuter_results))
                   for c in claims]
    logger.record(run_id, "judge_results", {"verdicts": [v.__dict__ for v in all_verdicts]})

    # 10. Synthesise
    synthesiser = OutputSynthesiser()
    leakage_flags = [r.leakage_flag for r in refuter_results]
    output = synthesiser.synthesise(query, claims, all_verdicts, leakage_flags)

    latency_ms = int((time.time() - start_time) * 1000)
    logger.finish_run(run_id, output.__dict__, latency_ms)
    return output

import time
import asyncio
import threading
from dotenv import load_dotenv

load_dotenv()

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


def run_coroutine_sync(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = {}
    error = {}

    def worker():
        try:
            result["value"] = asyncio.run(coro)
        except Exception as exc:
            error["value"] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result.get("value")

def get_refuter_result(claim: Claim, refuter_results: list):
    for r in refuter_results:
        if r.claim_id == claim.claim_id:
            return r
    return None


def _base_metadata() -> dict:
    return {
        "total_claims": 0,
        "supported": 0,
        "refuted": 0,
        "conflicting": 0,
        "unverifiable": 0,
        "leakage_flags": False,
        "route": "PENDING",
        "route_reason": "",
        "retrieval_reason": "",
        "abstention_reason": "",
        "loop_executed": False,
        "loop_reason": "",
        "refuter_queue_size": 0,
        "bypass_queue_size": 0,
        "pipeline_trace": [],
        "claim_explanations": [],
    }


def _add_trace(metadata: dict, stage: str, status: str, reason: str, **details):
    metadata["pipeline_trace"].append(
        {
            "stage": stage,
            "status": status,
            "reason": reason,
            "details": details,
        }
    )

def run_query(query: str, config=None, index: VectorIndex = None) -> SystemOutput:
    if config is None:
        config = load_config()
    if index is None:
        index = VectorIndex()
        
    logger = PipelineLogger(getattr(config.logging, "run_log_path", "logs/run_log.jsonl"))
    run_id = logger.start_run(query, route="pending")
    start_time = time.time()
    metadata = _base_metadata()
    
    # 1. Classify
    classifier = ClassifierAgent(config)
    route = classifier.classify(query)
    logger.record(run_id, "route", {"route": route})
    metadata["route"] = route
    
    if run_id in logger._runs:
        logger._runs[run_id]["route"] = route

    if route == "SKIP":
        proposer = ProposerAgent(config)
        try:
            answer_text = proposer.llm.invoke(query).content
        except Exception:
            answer_text = "Sorry, I cannot answer that right now."

        metadata["route_reason"] = "Classifier selected SKIP, so the query was answered directly without retrieval or refutation."
        _add_trace(metadata, "classifier", "skip", metadata["route_reason"], route=route)
            
        output = SystemOutput(
            natural_language_answer=answer_text,
            overall_confidence=1.0,
            claim_table=None,
            metadata=metadata
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
    metadata["retrieval_reason"] = f"Retrieved {len(passages)} passages; max similarity {max_sim:.2f}; abstention threshold {abs_thresh:.2f}."
    _add_trace(metadata, "retrieval", "ok", metadata["retrieval_reason"], passage_count=len(passages), max_similarity=max_sim, abstention_threshold=abs_thresh)
    if max_sim < abs_thresh:
        metadata["abstention_reason"] = f"Max similarity {max_sim:.2f} is below abstention threshold {abs_thresh:.2f}, so the pipeline stopped before claim decomposition/refutation."
        _add_trace(metadata, "abstention", "blocked", metadata["abstention_reason"], max_similarity=max_sim, abstention_threshold=abs_thresh)
        output = SystemOutput(
            natural_language_answer="Insufficient evidence in knowledge base to answer reliably.",
            overall_confidence=0.0,
            claim_table=None,
            metadata=metadata
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
    _add_trace(metadata, "decomposition", "ok", f"Decomposed answer into {len(claims)} validated claims.", claim_count=len(claims))

    if not claims:
        metadata["loop_reason"] = "No loop was run because claim decomposition produced zero validated claims."
        _add_trace(metadata, "screening", "skipped", metadata["loop_reason"], claim_count=0)
        output = SystemOutput(
            natural_language_answer=answer_text + "\n\n[This answer could not be verified.]",
            overall_confidence=0.0,
            claim_table=None,
            metadata=metadata
        )
        latency_ms = int((time.time() - start_time) * 1000)
        logger.finish_run(run_id, output.__dict__, latency_ms)
        return output

    # 6. Screen
    screener = ConfidenceScreener(config)
    refuter_queue, bypass_queue = screener.screen(claims)
    llm_threshold = getattr(config.screening, "llm_confidence_threshold", 0.85)
    faiss_threshold = getattr(config.screening, "faiss_similarity_threshold", 0.65)

    claim_explanations = []
    refuter_reasons = []
    for c in claims:
        reasons = []
        if c.llm_confidence < llm_threshold:
            reasons.append(f"LLM confidence {c.llm_confidence:.2f} < {llm_threshold:.2f}")
        if c.max_passage_similarity < faiss_threshold:
            reasons.append(f"FAISS similarity {c.max_passage_similarity:.2f} < {faiss_threshold:.2f}")
        if c.spot_check:
            reasons.append(f"spot-checked at rate {getattr(config.screening, 'spot_check_rate', 0.10):.2f}")

        route_label = "LOW_CONF" if c in refuter_queue else "HIGH_CONF"
        routing_reason = "; ".join(reasons) if reasons else "No screening trigger fired; claim bypassed refuter."
        claim_explanations.append(
            {
                "claim_id": c.claim_id,
                "claim_text": c.claim_text,
                "routing": route_label,
                "llm_confidence": c.llm_confidence,
                "max_passage_similarity": c.max_passage_similarity,
                "spot_check": c.spot_check,
                "reason": routing_reason,
            }
        )
        if c in refuter_queue:
            refuter_reasons.append({"claim_id": c.claim_id, "reason": routing_reason})

    metadata["claim_explanations"] = claim_explanations
    metadata["refuter_queue_size"] = len(refuter_queue)
    metadata["bypass_queue_size"] = len(bypass_queue)

    if refuter_queue:
        metadata["loop_executed"] = True
        metadata["loop_reason"] = (
            f"Refuter ran for {len(refuter_queue)} claim(s) because at least one claim was low-confidence or spot-checked."
        )
    else:
        metadata["loop_executed"] = False
        metadata["loop_reason"] = (
            "No loop was run because no claims were below the screening thresholds and no spot-check fired."
        )

    _add_trace(
        metadata,
        "screening",
        "ok",
        metadata["loop_reason"],
        refuter_queue_size=len(refuter_queue),
        bypass_queue_size=len(bypass_queue),
        refuter_reasons=refuter_reasons,
    )

    # 7. Refute (async)
    refuter = RefuterAgent(config)
    # Run refuter for screened claims in both LITE and FULL routes.
    # LITE still saves cost by routing fewer claims into `refuter_queue`.
    refuter_results = run_coroutine_sync(refuter.challenge_all(refuter_queue, index)) if refuter_queue else []
    if refuter_queue:
        _add_trace(metadata, "refutation", "ok", f"Refuter executed for {len(refuter_queue)} claims.", refuter_queue_size=len(refuter_queue))
    else:
        _add_trace(metadata, "refutation", "skipped", "Refuter was not executed because the screening queue was empty.", refuter_queue_size=0)
    
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
    _add_trace(metadata, "judging", "ok", f"Judge produced {len(all_verdicts)} verdict(s).", verdict_count=len(all_verdicts))

    # 10. Synthesise
    synthesiser = OutputSynthesiser()
    leakage_flags = [r.leakage_flag for r in refuter_results]
    output = synthesiser.synthesise(query, claims, all_verdicts, leakage_flags)

    # Keep verdict summary counts from synthesiser and append pipeline diagnostics.
    summary_metadata = dict(output.metadata or {})
    summary_metadata.update(
        {
            "route": metadata.get("route", "PENDING"),
            "route_reason": metadata.get("route_reason", ""),
            "retrieval_reason": metadata.get("retrieval_reason", ""),
            "abstention_reason": metadata.get("abstention_reason", ""),
            "loop_executed": bool(refuter_queue),
            "loop_reason": metadata.get("loop_reason", ""),
            "refuter_queue_size": len(refuter_queue),
            "bypass_queue_size": len(bypass_queue),
            "pipeline_trace": metadata.get("pipeline_trace", []),
            "claim_explanations": metadata.get("claim_explanations", []),
        }
    )
    output.metadata = summary_metadata

    latency_ms = int((time.time() - start_time) * 1000)
    logger.finish_run(run_id, output.__dict__, latency_ms)
    return output

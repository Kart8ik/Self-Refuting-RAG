# Software Architecture Document
## SR-RAG: Self-Refuting Retrieval Augmented Generation

---

**Document Version:** 1.0  
**Date:** March 2026  
**Team:** Team 20  
**Course:** Generative AI — PES University  

---

## Table of Contents

1. Introduction
2. Architectural Goals and Constraints
3. System Overview
4. Component Architecture
5. Data Flow and Sequence
6. Module Specifications
7. Prompt Architecture
8. Technology Stack
9. Deployment View
10. Key Design Decisions and Rationale
11. Risk Register

---

## 1. Introduction

### 1.1 Purpose

This Software Architecture Document (SAD) describes the internal structure, component responsibilities, data flows, technology choices, and design rationale for the SR-RAG system. It is intended to guide implementation and serve as a reference for understanding why the system is built the way it is.

### 1.2 Architectural Philosophy

SR-RAG is designed around three principles derived from the literature review:

**Adversarial asymmetry.** Borrowed from DRAG (Hu et al., ACL 2025): the Proposer agent has access to both retrieved documents and parametric LLM knowledge; the Refuter agent is constrained to retrieved documents only. This asymmetry ensures that any challenge the Refuter raises is grounded in documentary evidence, not hallucinated counter-argument.

**Claim-level granularity.** Unlike DRAG and MADAM-RAG which debate at the whole-answer level, SR-RAG decomposes answers into atomic claims before any verification occurs. This makes every verdict, confidence score, and evidence citation traceable to a specific fact rather than a vague answer-level judgement.

**Selective adversarial targeting.** Borrowed from CRAG (Yan et al., 2024): not every claim warrants a full adversarial retrieval pass. A confidence pre-screening step routes only low-confidence claims to the Refuter, controlling computational cost while preserving the full pipeline for claims that genuinely need it.

---

## 2. Architectural Goals and Constraints

| Goal | Description |
|------|-------------|
| Training-free | All agents operate via prompt engineering only — no fine-tuning |
| Modularity | Each agent is independently replaceable without modifying others |
| Reproducibility | Fixed seeds, versioned prompts, logged parameters |
| Transparency | Every claim has a full evidence chain to source documents |
| Cost efficiency | Selective Refuter targeting limits LLM API call count |
| Parallel execution | Refuter retrieval calls run in parallel across claims |

| Constraint | Impact |
|-----------|--------|
| Same LLM for all agents | Controls for model quality in evaluation; simplifies implementation |
| No parametric knowledge for Refuter | Enforced via system prompt; not technically verifiable |
| FAISS in-memory index | Sufficient for course project scale; not production-grade |
| Groq API rate limits | Evaluation subset limited to ~500 queries per dataset |

---

## 3. System Overview

SR-RAG consists of six processing stages arranged in a sequential pipeline with one parallel sub-stage (Refuter retrieval). A pre-pipeline query classifier gates the entire system.

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Classifier│ ──► SKIP ──► Direct LLM Answer
└────────┬────────┘
         │ LITE / FULL
         ▼
┌─────────────────┐
│    Proposer     │  (retrieved docs + parametric knowledge)
│    Agent        │  → generates answer → decomposes into claims
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Confidence   │  → HIGH_CONF claims bypass Refuter
│    Screener     │  → LOW_CONF claims → Refuter (parallel)
└────────┬────────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
HIGH_CONF   REFUTER AGENT  (retrieved docs only, parallel per-claim)
claims      └──► per-claim counter-evidence
    │             │
    └──────┬───────┘
           ▼
┌─────────────────┐
│   Judge Agent   │  → per-claim verdict + confidence score
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output Synth.   │  → NL answer + claim evidence table
└─────────────────┘
```

---

## 4. Component Architecture

### 4.1 Component Inventory

| Component | Type | Responsibility |
|-----------|------|---------------|
| `QueryClassifier` | LLM chain | Routes query to SKIP / LITE / FULL |
| `VectorIndex` | FAISS/HNSW | Stores and retrieves document embeddings |
| `EmbeddingModel` | Sentence Transformer | Encodes queries and documents to vectors |
| `ProposerAgent` | LangChain LLM chain | Generates answer with parametric + retrieved knowledge |
| `ClaimDecomposer` | Custom Python + LLM | Decomposes answer into atomic verifiable claims |
| `ConfidenceScreener` | Custom Python | Scores and routes claims by confidence |
| `RefuterAgent` | LangChain LLM chain | Adversarial retrieval; retrieved-docs-only mode |
| `JudgeAgent` | LangChain LLM chain | Resolves conflicts; assigns verdicts and confidence |
| `OutputSynthesiser` | Custom Python | Renders NL answer + claim evidence table |
| `EvaluationRunner` | Custom Python | Benchmark evaluation against baselines |

### 4.2 Component Dependency Graph

```
QueryClassifier
    └── LLMClient

ProposerAgent
    ├── LLMClient
    ├── VectorIndex
    └── EmbeddingModel

ClaimDecomposer
    └── LLMClient

ConfidenceScreener
    └── (pure Python logic, no LLM call)

RefuterAgent
    ├── LLMClient
    ├── VectorIndex
    └── EmbeddingModel

JudgeAgent
    └── LLMClient

OutputSynthesiser
    └── (pure Python, no LLM call)
```

---

## 5. Data Flow and Sequence

### 5.1 Full SR-RAG Query — Step by Step

**Step 1 — Classification**
```
INPUT:  query_text (str)
PROCESS: Single LLM call with classification prompt
OUTPUT: route ∈ {SKIP, LITE, FULL}
```

**Step 2 — Proposer Retrieval**
```
INPUT:  query_text
PROCESS: embed query → cosine search FAISS index → top-5 passages
OUTPUT: retrieved_passages: List[{doc_id, text, score}]
```

**Step 3 — Proposer Generation**
```
INPUT:  query_text + retrieved_passages
PROCESS: LLM call with Proposer system prompt
         (parametric knowledge permitted)
OUTPUT: answer_text (str)
```

**Step 4 — Claim Decomposition**
```
INPUT:  answer_text + retrieved_passages (for grounding reference)
PROCESS: LLM call with decomposition prompt
OUTPUT: claims: List[{
            claim_id, claim_text, confidence (float),
            supporting_doc_ids, routing (HIGH_CONF | LOW_CONF)
        }]
```

**Step 5 — Confidence Screening**
```
INPUT:  claims list
PROCESS: threshold(confidence < 0.85) → LOW_CONF
         10% random sample of HIGH_CONF → also routed to Refuter
OUTPUT: refuter_queue: List[claim]
        bypass_queue: List[claim]  (direct to Judge)
```

**Step 6 — Refuter Adversarial Retrieval (parallel)**
```
For each claim in refuter_queue (all calls issued concurrently):
    INPUT:  claim_text + supporting_doc_ids
    PROCESS: formulate adversarial query →
             embed → FAISS search →
             LLM call (retrieved-docs-only prompt)
    OUTPUT: refuter_result: {
                verdict (CONTESTED | INSUFFICIENT | NOT_FOUND),
                counter_passages: List[{doc_id, text}],
                query_used: str
            }
```

**Step 7 — Judge Resolution**
```
For each claim (bypass_queue + refuter_queue results):
    INPUT:  claim + proposer_evidence + refuter_result
    PROCESS: LLM call with Judge rubric prompt
    OUTPUT: judge_result: {
                verdict (SUPPORTED | REFUTED | CONFLICTING | UNVERIFIABLE),
                final_confidence (float),
                justification (str),
                supporting_evidence: List[passage],
                counter_evidence: List[passage]
            }
```

**Step 8 — Output Synthesis**
```
INPUT:  all judge_results + original query
PROCESS: filter REFUTED claims from answer body
         flag CONFLICTING claims inline
         compute overall confidence (mean, with penalties)
         render claim evidence table if CONFLICTING/REFUTED exist
OUTPUT: {
    natural_language_answer (str),
    overall_confidence (float),
    claim_table: List[claim_row],  (only if needed)
    metadata: {total_claims, supported, refuted, conflicting, unverifiable}
}
```

### 5.2 Sequence Diagram (Textual)

```
User → QueryClassifier: query
QueryClassifier → LLM: classify(query)
LLM → QueryClassifier: FULL
QueryClassifier → ProposerAgent: query

ProposerAgent → VectorIndex: retrieve(query, k=5)
VectorIndex → ProposerAgent: passages[]
ProposerAgent → LLM: generate_answer(query, passages)
LLM → ProposerAgent: answer_text
ProposerAgent → ClaimDecomposer: decompose(answer_text, passages)
ClaimDecomposer → LLM: extract_claims(answer_text)
LLM → ClaimDecomposer: claims[]
ClaimDecomposer → ConfidenceScreener: claims[]

ConfidenceScreener → RefuterAgent: low_conf_claims[] (parallel)
ConfidenceScreener → JudgeAgent: high_conf_claims[] (direct)

RefuterAgent → VectorIndex: adversarial_retrieve(claim) [×N parallel]
VectorIndex → RefuterAgent: counter_passages[]
RefuterAgent → LLM: evaluate_counter_evidence(claim, counter_passages)
LLM → RefuterAgent: refuter_result

RefuterAgent → JudgeAgent: (claim, refuter_result)
JudgeAgent → LLM: resolve(claim, proposer_evidence, refuter_result)
LLM → JudgeAgent: verdict + confidence + justification

JudgeAgent → OutputSynthesiser: all_verdicts[]
OutputSynthesiser → User: {answer, confidence, claim_table}
```

---

## 6. Module Specifications

### 6.1 QueryClassifier

```python
class QueryClassifier:
    """
    Routes incoming queries to appropriate pipeline level.
    Single LLM call. Returns one of {SKIP, LITE, FULL}.
    """
    def classify(self, query: str) -> Literal["SKIP", "LITE", "FULL"]:
        ...
```

**Routing logic:**
- SKIP: factual consensus, personal/subjective, conversational, creative
- LITE: simple single-hop factual lookup (one source sufficient)
- FULL: multi-hop, contested topics, medical/legal/scientific, sourced evidence required

### 6.2 VectorIndex

```python
class VectorIndex:
    """
    FAISS HNSW index over chunked corpus documents.
    Handles build, save, load, and retrieve operations.
    """
    def build(self, documents: List[str], metadata: List[dict]) -> None: ...
    def retrieve(self, query: str, k: int = 5) -> List[RetrievedPassage]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...
```

**Index parameters:**
- Algorithm: HNSW (M=16, efConstruction=200, efSearch=50)
- Distance metric: Inner product (L2-normalised vectors → cosine similarity)
- Chunk size: 512 tokens, 128-token overlap
- Embedding dimension: 384

### 6.3 ProposerAgent

```python
class ProposerAgent:
    """
    Generates initial answer using retrieved docs + parametric knowledge.
    Also triggers claim decomposition.
    """
    def generate(self, query: str, passages: List[RetrievedPassage]
                ) -> tuple[str, List[Claim]]: ...
```

**System prompt constraints:**
- Instructed to ground answers in retrieved documents
- Permitted to use background knowledge to add depth and coherence
- Instructed to be explicit about uncertainty

### 6.4 ClaimDecomposer

```python
class ClaimDecomposer:
    """
    Decomposes answer text into atomic, self-contained, verifiable claims.
    Assigns initial confidence score to each claim.
    """
    def decompose(self, answer: str, passages: List[RetrievedPassage]
                 ) -> List[Claim]: ...
```

**Decomposition rules enforced by prompt:**
1. One fact per claim
2. All entities named explicitly (no pronouns)
3. Claim must be falsifiable from a document
4. Vague/hedged statements are discarded, not decomposed
5. Confidence based on how strongly retrieved passages support the claim

### 6.5 ConfidenceScreener

```python
class ConfidenceScreener:
    """
    Pure Python. Routes claims to Refuter or bypass.
    No LLM call.
    """
    THRESHOLD = 0.85
    SPOT_CHECK_RATE = 0.10

    def screen(self, claims: List[Claim]
              ) -> tuple[List[Claim], List[Claim]]:
        """Returns (refuter_queue, bypass_queue)"""
        ...
```

### 6.6 RefuterAgent

```python
class RefuterAgent:
    """
    Adversarial retrieval and challenge per claim.
    STRICTLY retrieved-document-only mode.
    """
    async def challenge(self, claim: Claim,
                       index: VectorIndex) -> RefuterResult: ...

    async def challenge_all(self, claims: List[Claim],
                           index: VectorIndex) -> List[RefuterResult]: ...
        # Issues all challenge() calls concurrently via asyncio.gather()
```

**System prompt constraints (critical):**
- "You must ONLY use the documents provided below. Do not use any knowledge from your training."
- "Every argument you make must cite a specific document ID and passage."
- "If no document supports a counter-argument, output NOT_FOUND."

### 6.7 JudgeAgent

```python
class JudgeAgent:
    """
    Resolves per-claim conflicts. Assigns verdict and confidence.
    """
    def judge_claim(self, claim: Claim,
                   proposer_evidence: List[Passage],
                   refuter_result: RefuterResult) -> Verdict: ...

    def compute_overall_confidence(self, verdicts: List[Verdict]) -> float:
        """Mean confidence; CONFLICTING at 0.5 weight; REFUTED at 0."""
        ...
```

**Confidence scoring axes:**
- Evidence relevance (weight 0.40): avg cosine similarity of supporting passages
- Evidence count (weight 0.35): number of independent corroborating documents
- Claim specificity (weight 0.25): penalty for vague claims (LLM-scored 0–1)

### 6.8 OutputSynthesiser

```python
class OutputSynthesiser:
    """
    Renders final answer and optional claim evidence table.
    Pure Python, no LLM call.
    """
    def synthesise(self, query: str,
                  verdicts: List[Verdict]) -> SystemOutput: ...
```

**Output rules:**
- SUPPORTED claims → included in answer body, stated normally
- CONFLICTING claims → included with `[⚠ conflicting evidence]` marker
- REFUTED claims → omitted from answer; noted as "one claim could not be verified"
- UNVERIFIABLE claims → included with "could not be verified from available sources"
- Claim evidence table rendered only when CONFLICTING or REFUTED claims exist

---

## 7. Prompt Architecture

All prompts are versioned in `prompts/v1/`. Each agent has a separate system prompt file. Prompts are loaded at runtime via a `PromptRegistry` class. This ensures reproducibility and allows prompt ablation experiments.

### 7.1 Prompt Files

```
prompts/
  v1/
    classifier_system.txt
    proposer_system.txt
    decomposer_system.txt
    refuter_system.txt       ← critical: retrieved-docs-only instruction
    judge_system.txt
```

### 7.2 Decomposer Prompt (Core Logic)

```
You are a claim extraction assistant. Given an answer text,
extract every verifiable factual claim it contains.

Rules:
1. Each claim must be ONE fact only.
2. Resolve all pronouns — use the full entity name.
3. Only include claims that are factual and falsifiable.
4. Discard vague, hedged, or opinion statements.
5. For each claim, score your confidence (0.0–1.0) based on
   how strongly the provided source passages support it.

Output JSON array:
[{"claim_text": "...", "confidence": 0.0, "supporting_doc_ids": ["..."]}]
```

### 7.3 Refuter Prompt (Core Logic)

```
You are an adversarial fact-checker.
Your ONLY knowledge source is the documents provided below.
Do NOT use anything you know from training.

Given the claim: "{claim_text}"

Search the documents for ANY evidence that contradicts,
qualifies, or complicates this claim.

If you find contradicting evidence: output CONTESTED with the relevant passages.
If documents are inconclusive: output INSUFFICIENT.
If no relevant documents exist: output NOT_FOUND.

Cite document IDs for every argument you make.
```

### 7.4 Judge Prompt (Core Logic)

```
You are an impartial evidence judge.
Given a claim, the proposer's supporting evidence, and the refuter's
counter-evidence, assign one verdict:

SUPPORTED — proposer evidence is clear; refuter found nothing meaningful
REFUTED — refuter's evidence clearly outweighs proposer's
CONFLICTING — both sides have real evidence; neither clearly wins
UNVERIFIABLE — no relevant documents found on either side

Also assign a confidence score (0.0–1.0) and a one-sentence justification.

Output JSON: {"verdict": "...", "confidence": 0.0, "justification": "..."}
```

---

## 8. Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| LLM | Llama 3.3 70B (Groq API) | Strong reasoning; fast inference on Groq; free tier; consistent across agents |
| Embedding | all-MiniLM-L6-v2 (sentence-transformers) | Lightweight, fast, good quality for semantic search |
| Vector index | FAISS IndexHNSWFlat | Approximate NN at scale; in-memory; well-documented |
| Orchestration | LangChain v0.3 | Agent chains, prompt templates, retrieval chains |
| Custom logic | Python 3.10+ | Claim decomposition, confidence screening, output rendering |
| Async execution | `asyncio` + `aiohttp` | Parallel Refuter retrieval calls |
| Evaluation | HuggingFace `datasets`, custom scripts | Standard benchmark loading and metric computation |
| Data | JSON, .faiss binary | Claim schemas, index persistence |

---

## 9. Deployment View

SR-RAG is deployed as a local Python application for the course project. There is no web interface or server. The system is invoked via a command-line interface for both interactive queries and evaluation runs.

```
sr_rag/
├── agents/
│   ├── proposer.py
│   ├── refuter.py
│   ├── judge.py
│   └── classifier.py
├── pipeline/
│   ├── claim_decomposer.py
│   ├── confidence_screener.py
│   └── output_synthesiser.py
├── retrieval/
│   ├── vector_index.py
│   └── embedding_model.py
├── prompts/
│   └── v1/
│       ├── classifier_system.txt
│       ├── proposer_system.txt
│       ├── decomposer_system.txt
│       ├── refuter_system.txt
│       └── judge_system.txt
├── evaluation/
│   ├── runner.py
│   ├── metrics.py
│   └── baselines/
│       ├── vanilla_rag.py
│       ├── self_rag.py
│       └── drag.py
├── data/
│   ├── index.faiss
│   └── index_metadata.json
├── config.yaml
├── main.py
└── requirements.txt
```

---

## 10. Key Design Decisions and Rationale

### Decision 1 — Claim-level granularity over answer-level debate

**Alternatives considered:** DRAG-style whole-answer debate; sentence-level debate  
**Decision:** Atomic claim-level decomposition  
**Rationale:** Whole-answer debate (DRAG, MADAM-RAG) produces a single verdict for an entire answer, which may contain a mix of correct and incorrect facts. Claim-level granularity allows partial answers — some claims supported, others conflicting — to be expressed honestly. It also enables precise evidence tracing: each verdict links to a specific source passage for a specific fact.

### Decision 2 — Selective Refuter targeting (confidence gating)

**Alternatives considered:** Challenge all claims; challenge none (Judge-only)  
**Decision:** Challenge claims below 0.85 confidence + 10% spot-check of high-confidence  
**Rationale:** MAD Strategies (Smit et al., ICML 2024) shows multi-agent debate is expensive and doesn't always outperform simpler methods. Targeting only low-confidence claims reduces API call count by ~60–70% for typical well-retrieved answers while preserving adversarial coverage for claims that need it. The spot-check prevents the system from silently trusting all high-confidence claims.

### Decision 3 — Refuter in retrieved-document-only mode

**Alternatives considered:** Full LLM access for Refuter; retrieval-only (no LLM)  
**Decision:** LLM + retrieved docs only (no parametric knowledge)  
**Rationale:** Directly adopted from DRAG's asymmetric information design. If the Refuter can use parametric knowledge, it could hallucinate counter-arguments that sound convincing but have no documentary basis. Constraining it to retrieved documents ensures every challenge is citable. Implementation is prompt-level only.

### Decision 4 — Same LLM for all agents

**Alternatives considered:** Speculative RAG style (small model for Refuter, large for Judge)  
**Decision:** Single model (Llama 3.3 70B) for all agents  
**Rationale:** Using different model sizes introduces model capability as a confound in evaluation. For a course project, controlling this variable matters more than the latency gains. Mixed-model approach is noted as a future work direction.

### Decision 5 — CONFLICTING verdict surfaces both sides to user

**Alternatives considered:** Force Judge to pick one side; suppress conflicting claims  
**Decision:** Expose both sides with evidence citations  
**Rationale:** In high-stakes domains (medical, legal), forcing a resolution where genuine conflict exists is more dangerous than surfacing uncertainty. No existing paper in the literature review does this explicitly — it is SR-RAG's primary transparency contribution.

---

## 11. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Refuter leaks parametric knowledge despite prompt constraint | Medium | Medium | Include explicit "forbidden knowledge" test in unit tests |
| Claim decomposition produces vague or redundant claims | High | High | Invest in prompt tuning; run decomposition-only ablation first |
| Groq API rate limit hit during evaluation | Medium | Medium | Batch requests; use subset of 500 questions; implement retry logic |
| FAISS index quality insufficient for adversarial queries | Low | High | Test retrieval recall on known queries before full evaluation run |
| MAD Strategies finding: debate doesn't reliably beat self-consistency | Medium | Medium | Confirm this with ablation; ensure SR-RAG's claim-level design addresses the hyperparameter sensitivity finding |
| Judge cannot reliably resolve conflicts (outputs inconsistent verdicts) | Medium | High | Define structured rubric; run Judge-only evaluation on a labelled set before full integration |

---

*End of SAD — SR-RAG v1.0*

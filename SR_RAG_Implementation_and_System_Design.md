# SR-RAG: Implementation and System Design

## 1. System Overview (High-Level)

SR-RAG (Self-Refuting Retrieval-Augmented Generation) is a multi-agent pipeline that augments standard RAG with an internal adversarial verification loop to detect and mitigate hallucinated or unsupported claims before they reach the end user. Given a user query, the system first classifies query complexity, retrieves relevant documents from a vector store, generates a draft answer grounded in those documents, then decomposes the answer into atomic factual claims. Each claim is independently assessed for confidence; low-confidence claims are routed to an adversarial refuter agent that searches for contradicting evidence using only the retrieved corpus. A judge agent arbitrates between supporting and counter-evidence to produce a per-claim verdict, and a synthesiser assembles the final user-facing response annotated with confidence metadata and a transparency table.

The major pipeline stages and their purposes are:

- **Query Classifier**: Determines query complexity and routes it to the appropriate processing depth (skip, lite, or full verification).
- **Vector Retrieval System**: Retrieves the top-*k* semantically similar passages from a FAISS index to ground the answer.
- **Proposer Agent**: Generates a draft natural-language answer conditioned on the query and retrieved passages.
- **Claim Decomposer**: Breaks the draft answer into atomic, independently verifiable factual claims with associated confidence scores.
- **Claim Validator**: Filters decomposed claims by removing vague, entity-free, or near-duplicate claims.
- **Confidence Screener**: Routes claims into a refutation queue or bypass queue based on dual confidence thresholds and stochastic spot-checking.
- **Refuter Agent**: Adversarially challenges low-confidence claims by retrieving counter-evidence and producing a structured verdict, constrained to use only retrieved documents.
- **Evidence Scorer**: Computes a programmatic confidence score for each claim based on retrieval relevance, source count, and claim specificity.
- **Judge Agent**: Arbitrates between proposer evidence and refuter counter-evidence to produce a final per-claim verdict (SUPPORTED, REFUTED, CONFLICTING, or UNVERIFIABLE).
- **Output Synthesiser**: Assembles the verified claims into a coherent final response with an overall confidence score, claim-level transparency table, and diagnostic metadata.

---

## 2. Data Flow

The data flows through the system in the following sequence:

1. **Query Input**: The user submits a natural-language query.

2. **Classification**: The Query Classifier assigns a route label — SKIP (no retrieval needed), LITE (single-hop factual), or FULL (multi-hop, contested, or domain-sensitive). If SKIP, the LLM answers directly and the pipeline terminates early.

3. **Retrieval**: The query is embedded using a sentence transformer and the top-*k* passages are retrieved from a FAISS inner-product index. Each passage carries a similarity score.

4. **Abstention Check**: If the maximum similarity score among all retrieved passages falls below a configurable abstention threshold (default 0.40), the system declines to answer, returning an abstention message with zero confidence.

5. **Answer Proposal**: The Proposer Agent generates a draft answer conditioned on the query and the retrieved passages.

6. **Claim Decomposition**: The draft answer is decomposed into a JSON array of atomic claims. Each raw claim carries an LLM-assigned confidence score and a list of supporting document identifiers.

7. **Claim Validation**: Raw claims are filtered through three validation gates: minimum word count, presence of a named entity or numeric value, and pairwise cosine deduplication. Only validated claims proceed.

8. **Confidence Screening**: Each validated claim is evaluated against two thresholds — LLM confidence (default 0.85) and maximum passage similarity (default 0.65). Claims failing either threshold are routed to the refuter queue (LOW_CONF). Among high-confidence claims, a stochastic spot-check (default 10% rate) randomly promotes a subset to the refuter queue as well. Remaining claims bypass refutation (HIGH_CONF).

9. **Adversarial Refutation**: For each claim in the refuter queue, the Refuter Agent formulates an adversarial retrieval query, retrieves counter-evidence from the same vector index, and prompts the LLM to produce one of three verdicts: CONTESTED, INSUFFICIENT, or NOT_FOUND. A leakage detection flag is raised if the refuter's output references document identifiers not present in the retrieved set. Claims are processed concurrently with a bounded semaphore.

10. **Evidence Scoring**: A programmatic evidence scorer computes a composite confidence for every claim (both refuted and bypassed) based on a weighted combination of retrieval relevance, source diversity, and claim specificity.

11. **Adjudication**: The Judge Agent receives the proposer evidence, refuter result (if any), and the programmatic evidence score bundle for each claim, and produces a structured verdict: SUPPORTED, REFUTED, CONFLICTING, or UNVERIFIABLE, along with a calibrated confidence and justification. Deterministic calibration guardrails override the LLM's verdict when it is inconsistent with the evidence profile.

12. **Synthesis**: The Output Synthesiser aggregates per-claim verdicts into a final natural-language answer. Supported claims are included verbatim; conflicting claims are annotated with warnings; refuted claims are omitted with a notice. A transparency table is generated for any claim that is conflicting, refuted, or below a confidence threshold, listing supporting and counter-evidence side by side. An overall confidence score is computed as the weighted mean of verdict-level scores.

---

## 3. Core Components and Responsibilities

### 3.1 Query Classifier
- **Input**: Raw user query (string).
- **Output**: One of three route labels: SKIP, LITE, or FULL.
- **Responsibility**: Determines whether the query requires retrieval and adversarial verification. Simple, conversational, or creative queries are routed to SKIP (direct LLM response). Single-hop factual queries use LITE, while multi-hop, contested, or domain-sensitive queries use FULL. Both LITE and FULL engage the retrieval-refutation loop; the screener naturally reduces refuter workload for LITE-style queries by routing fewer claims.
- **Constraints**: Defaults to FULL on any classification failure to ensure maximum safety.

### 3.2 Proposer Agent
- **Input**: User query and a list of retrieved passages.
- **Output**: A draft natural-language answer (string).
- **Responsibility**: Generates a grounded answer using retrieved documents as context. The proposer is instructed to be explicit about uncertainty and to ground claims in provided passages.
- **Constraints**: Raises an abstention error if no passages are provided or if all passages fall below the similarity threshold.

### 3.3 Claim Decomposer
- **Input**: The proposer's draft answer and the retrieved passages.
- **Output**: A list of structured Claim objects.
- **Responsibility**: Extracts every independently verifiable factual assertion from the draft answer. Each claim is a single fact with all pronouns resolved, accompanied by the LLM's self-assessed confidence score and the identifiers of supporting documents. The decomposer enforces strict JSON-only output formatting.
- **Constraints**: Discards vague, hedged, or opinion-based statements by instruction. Post-decomposition validation further filters claims.

### 3.4 Claim Validator
- **Input**: Raw claim dictionaries from the decomposer.
- **Output**: A filtered list of validated claim dictionaries and rejection statistics.
- **Responsibility**: Applies three validation gates: (1) minimum word count filter to reject trivially short claims, (2) named entity or numeric presence check to ensure claims are substantive and verifiable, and (3) pairwise cosine similarity deduplication to remove near-duplicate claims. When duplicates are detected, the longer (more specific) claim is retained.
- **Constraints**: Uses the same embedding model as the retrieval system for deduplication similarity computation.

### 3.5 Confidence Screener
- **Input**: A list of validated Claim objects.
- **Output**: Two lists — a refuter queue (claims requiring adversarial verification) and a bypass queue (claims deemed sufficiently confident).
- **Responsibility**: Implements a dual-threshold routing policy. A claim is routed to the refuter if either its LLM-assigned confidence falls below the LLM confidence threshold or its maximum supporting passage similarity falls below the FAISS similarity threshold. Additionally, a configurable fraction of high-confidence claims are stochastically spot-checked using a deterministic per-claim hash seed.
- **Constraints**: Spot-check randomness is deterministic per claim identifier and global seed, ensuring reproducibility.

### 3.6 Refuter Agent
- **Input**: A Claim object and access to the vector index.
- **Output**: A RefuterResult containing a verdict (CONTESTED, INSUFFICIENT, or NOT_FOUND), counter-passages, the adversarial query used, and a leakage flag.
- **Responsibility**: Acts as an adversarial fact-checker. Formulates a negation-oriented retrieval query ("evidence contradicting or disproving: [claim]"), retrieves counter-evidence from the same corpus, and prompts the LLM to evaluate whether the retrieved documents contradict, qualify, or are irrelevant to the claim.
- **Constraints**: (1) The refuter is strictly constrained to use only the documents provided in the retrieval set — it must not draw on parametric knowledge. If no retrieved document contradicts the claim, the verdict must be NOT_FOUND. (2) A leakage detection mechanism flags cases where the refuter's output references document identifiers absent from the retrieved set, indicating potential hallucination of evidence. (3) Concurrent execution is bounded by an asyncio semaphore (default cap of 3).

### 3.7 Evidence Scorer
- **Input**: A Claim, the proposer passages, and optionally a RefuterResult.
- **Output**: An EvidenceScoreBundle with a composite programmatic confidence.
- **Responsibility**: Computes a purely programmatic (non-LLM) confidence score via a weighted linear combination of three sub-scores: (1) **Relevance** (weight 0.40) — the mean similarity score of passages that the claim cites as supporting evidence; (2) **Count** (weight 0.35) — the number of unique source titles among supporting passages, normalised to a ceiling of 3; (3) **Specificity** (weight 0.25) — a binary indicator of whether the claim contains numeric data or proper nouns, proxying for falsifiability.

### 3.8 Judge Agent
- **Input**: A Claim, the EvidenceScoreBundle, proposer passages, and optionally a RefuterResult.
- **Output**: A JudgeVerdict with verdict, calibrated confidence, justification, and evidence references.
- **Responsibility**: Serves as the final arbiter for each claim. Receives both the proposer's supporting evidence and the refuter's counter-evidence (if any), along with programmatic scores, and produces one of four verdicts: SUPPORTED, REFUTED, CONFLICTING, or UNVERIFIABLE. The judge's LLM-derived verdict undergoes deterministic calibration: if the refuter found contesting evidence but the LLM labelled the claim SUPPORTED, the verdict is overridden to CONFLICTING or REFUTED depending on evidence strength. The final confidence is the average of the calibrated LLM confidence and the programmatic confidence. A heuristic fallback produces verdicts deterministically when LLM calls fail.
- **Constraints**: Calibration guardrails prevent the LLM from dismissing refuter evidence. Strong or absolute claims (containing terms like "always," "never," "robust," "guarantee") face a stricter relevance threshold for SUPPORTED verdicts.

### 3.9 Output Synthesiser
- **Input**: The original query, all claims, all judge verdicts, and leakage flags.
- **Output**: A SystemOutput containing a natural-language answer, overall confidence, a claim transparency table, and metadata.
- **Responsibility**: Assembles the final response. Supported claims are included verbatim. Conflicting claims are annotated with a warning and referenced in the claim table. Refuted claims are omitted from the answer body with a user-facing notice. The claim table includes per-claim verdicts, confidence scores, and representative supporting and counter-evidence excerpts. Overall confidence is computed as the weighted mean of verdict-level scores (SUPPORTED = 1.0, CONFLICTING = 0.5, REFUTED = 0.0, UNVERIFIABLE = 0.0).

### 3.10 Vector Retrieval System
- **Input**: A query string and the number of passages to retrieve (*k*).
- **Output**: A ranked list of RetrievedPassage objects with similarity scores.
- **Responsibility**: Maintains a FAISS index over L2-normalised dense embeddings produced by a sentence transformer (all-MiniLM-L6-v2, 384 dimensions). Supports both flat inner-product search and optional HNSW approximate search. Serves both the proposer (original query retrieval) and the refuter (adversarial query retrieval) stages.

---

## 4. Key Design Mechanisms

### 4.1 Claim-Level Decomposition
The system decomposes the proposer's draft answer into atomic factual claims, each containing exactly one verifiable assertion. A "claim" is defined as a self-contained factual statement with all coreferences resolved, accompanied by a confidence estimate and pointers to supporting documents. This decomposition is essential because it enables independent verification of each assertion, preventing a single hallucinated fact from contaminating an otherwise correct response. It also enables selective routing: only uncertain claims need adversarial scrutiny.

### 4.2 Confidence-Based Routing
Claims are routed to the refuter based on a dual-threshold policy: LLM self-assessed confidence (default threshold 0.85) and maximum FAISS similarity of supporting passages (default threshold 0.65). A claim must exceed both thresholds to bypass refutation. This dual-signal design combines the LLM's intrinsic uncertainty estimation with an external, retrieval-based grounding signal, reducing the risk of overconfident but poorly grounded claims escaping verification.

### 4.3 Selective Adversarial Verification
Not all claims are sent to the refuter — only those flagged as low-confidence or randomly spot-checked. This selective routing exists to balance verification thoroughness against computational cost and latency. The stochastic spot-check mechanism (default 10% of high-confidence claims) provides statistical coverage of the bypass queue, ensuring that even confidently-stated claims are periodically subjected to adversarial scrutiny. The spot-check is deterministic per claim identifier to preserve reproducibility.

### 4.4 Refuter Constraint (Retrieved-Documents-Only)
The refuter agent is explicitly instructed that its sole knowledge source is the set of documents retrieved for the adversarial query. It must not draw on parametric knowledge from training. If no retrieved document contains contradicting evidence, the refuter must output NOT_FOUND rather than fabricating a counter-argument. This constraint is critical for preventing the adversarial agent itself from hallucinating counter-evidence, which would undermine the entire verification loop. A leakage detection mechanism further enforces this constraint by flagging outputs that reference document identifiers absent from the retrieved set.

### 4.5 Judge Decision Logic
The judge resolves conflicts between proposer evidence and refuter counter-evidence through a layered decision process. First, an LLM produces a verdict and confidence based on the full evidence context and programmatic scores. Second, deterministic calibration guardrails override the LLM's verdict when it is inconsistent with the evidence profile — for instance, a claim cannot remain SUPPORTED if the refuter found contesting evidence, and strong/absolute claims face a stricter retrieval relevance threshold. Third, the final confidence is computed as the average of the calibrated LLM confidence and the programmatic evidence score, anchoring the output to both neural and retrieval-based signals. When LLM calls fail entirely, a heuristic fallback produces verdicts purely from the programmatic confidence and refuter verdict, ensuring graceful degradation.

---

## 5. Data Structures

### 5.1 Claim
| Field | Type | Description |
|---|---|---|
| claim_id | string | Unique identifier scoped to the pipeline run |
| claim_text | string | The atomic factual assertion with coreferences resolved |
| llm_confidence | float (0.0–1.0) | The decomposer LLM's self-assessed confidence that the claim is supported by the retrieved passages |
| max_passage_similarity | float | The highest FAISS similarity score among passages cited as supporting this claim |
| supporting_doc_ids | list of strings | Identifiers of documents the LLM cited as supporting this claim |
| routing | string | Classification label — HIGH_CONF or LOW_CONF — assigned by the screener |
| spot_check | boolean | Whether this high-confidence claim was selected for stochastic adversarial verification |

### 5.2 Refuter Result
| Field | Type | Description |
|---|---|---|
| claim_id | string | The claim being challenged |
| verdict | string | One of CONTESTED, INSUFFICIENT, or NOT_FOUND |
| counter_passages | list of passages | Passages cited as counter-evidence, each with document identifier and text |
| query_used | string | The adversarial query formulated by the refuter for retrieval |
| leakage_flag | boolean | True if the refuter referenced document identifiers not present in its retrieved set |

### 5.3 Judge Verdict
| Field | Type | Description |
|---|---|---|
| claim_id | string | The claim being judged |
| verdict | string | One of SUPPORTED, REFUTED, CONFLICTING, or UNVERIFIABLE |
| final_confidence | float (0.0–1.0) | Average of calibrated LLM confidence and programmatic evidence confidence |
| justification | string | Natural-language explanation of the verdict, potentially including calibration notes |
| supporting_evidence | list of passages | Proposer passages that support this claim |
| counter_evidence | list of passages | Refuter passages that contest this claim |

### 5.4 System Output (Final Output Format)
| Field | Type | Description |
|---|---|---|
| natural_language_answer | string | The synthesised answer with annotations for conflicting or unverifiable claims and omission notices for refuted claims |
| overall_confidence | float (0.0–1.0) | Weighted average across all claim verdicts |
| claim_table | list of dicts or null | Transparency table for flagged claims containing claim text, verdict, confidence, and representative evidence excerpts |
| metadata | dictionary | Diagnostic information including claim counts by verdict, leakage flags, routing decisions, pipeline trace, and per-claim screening explanations |

---

## 6. Retrieval and Embedding

The retrieval subsystem is built on FAISS (Facebook AI Similarity Search) with dense embeddings produced by the `all-MiniLM-L6-v2` sentence transformer model, which produces 384-dimensional embeddings. Documents are L2-normalised prior to indexing, and retrieval uses inner-product similarity (equivalent to cosine similarity after normalisation). The system supports two index types: a flat exhaustive search index (IndexFlatIP, default) for stability and an optional HNSW approximate nearest neighbour index (IndexHNSWFlat) with configurable construction and search parameters (efConstruction=200, efSearch=50, M=16) for larger corpora.

The embedding model is implemented as a singleton with lazy loading to avoid unnecessary initialisation overhead. Documents are encoded in batches of 16 for throughput.

Retrieval serves two distinct roles in the pipeline:

1. **Proposer retrieval**: The original user query is used to retrieve the top-*k* (default *k*=5) passages to ground the draft answer.
2. **Refuter retrieval**: An adversarial query (prefixed with "evidence contradicting or disproving:") is used to retrieve passages that may contradict a specific claim from the same index.

Both retrieval operations share the same FAISS index and embedding model, ensuring a unified knowledge base and consistent similarity semantics.

The corpus can be loaded from Hugging Face datasets, local files (TXT, CSV, JSON, JSONL, Parquet), or user-uploaded text via the API. Uploaded documents are chunked using a paragraph-aware strategy with configurable chunk size (default 1200 characters) and overlap (default 150 characters).

---

## 7. Efficiency Considerations

Several mechanisms reduce computation and API cost:

1. **Query classification and early exit**: The classifier routes trivial queries to a direct LLM call (SKIP), entirely bypassing retrieval, decomposition, and verification. This eliminates multiple LLM calls and vector searches for queries that do not benefit from adversarial verification.

2. **Abstention threshold**: If the best retrieved passage has a similarity score below a configurable threshold (default 0.40), the pipeline terminates early with an abstention message, avoiding wasteful downstream LLM calls on queries outside the knowledge base's coverage.

3. **Selective refutation via confidence-based routing**: Only claims that fail confidence thresholds are routed to the refuter. High-confidence claims bypass adversarial verification entirely. This can substantially reduce the number of refuter LLM calls — in the best case, to zero.

4. **Concurrent asynchronous refutation**: Claims in the refuter queue are processed concurrently using asynchronous HTTP calls with a bounded semaphore (default cap of 3). This reduces wall-clock latency compared to sequential processing while preventing API rate-limit exhaustion.

5. **Singleton embedding model**: The sentence transformer is loaded once and shared across all retrieval operations, avoiding repeated model initialisation.

6. **Heuristic fallback in the judge**: When LLM calls fail (e.g., due to rate limiting), the judge produces verdicts deterministically from programmatic evidence scores rather than retrying indefinitely, ensuring bounded latency and graceful degradation.

7. **Claim validation and deduplication**: Filtering vague, entity-free, and duplicate claims before screening reduces the total number of claims entering the verification pipeline.

---

## 8. Summary of Novel Contributions (from Implementation Perspective)

1. **Claim-level adversarial verification within RAG**: Unlike standard RAG pipelines that treat the generated answer as a monolithic output, SR-RAG decomposes the answer into atomic claims and applies targeted adversarial scrutiny only where uncertainty is detected, enabling fine-grained hallucination detection and mitigation.

2. **Dual-signal confidence routing with stochastic spot-checking**: The screener combines LLM self-assessed confidence with retrieval similarity scores to make routing decisions, and augments this with a randomised spot-check mechanism to provide statistical coverage of the bypass queue — balancing thoroughness against cost.

3. **Retrieval-constrained adversarial refutation**: The refuter agent is architecturally constrained to use only documents from the retrieved corpus, preventing it from hallucinating counter-evidence from parametric knowledge. A leakage detection mechanism provides an additional integrity check on refuter outputs.

4. **Calibrated adjudication with deterministic guardrails**: The judge combines LLM-based reasoning with programmatic evidence scores and applies deterministic override rules to prevent verdict inflation (e.g., a claim cannot be labelled SUPPORTED if the refuter found contesting evidence), providing robustness beyond what a single LLM call can guarantee.

5. **Transparent claim-level output with provenance**: The final output includes a structured claim table with per-claim verdicts, confidence scores, and representative supporting and counter-evidence excerpts, enabling users to inspect the system's reasoning at the claim level rather than accepting a black-box answer.

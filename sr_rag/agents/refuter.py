import os
import json
import asyncio
import httpx
from typing import List
from sr_rag.config import load_config
from sr_rag.models import Claim, RefuterResult, RetrievedPassage
from sr_rag.retrieval.vector_index import VectorIndex

class RefuterAgent:
    def __init__(self, config=None):
        if config is None:
            self.config = load_config()
        else:
            self.config = config
            
        self.api_key = os.environ.get("GROQ_API_KEY", "dummy_key")
        
        prompt_path = os.path.join(
            getattr(self.config.prompts, "base_path", "prompts/"), 
            getattr(self.config.prompts, "version", "v1"), 
            "refuter_system.txt"
        )
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system_prompt_template = f.read().strip()
            
        self.k = getattr(self.config.retrieval, "k", 5)

    async def challenge(self, claim: Claim, index: VectorIndex, semaphore: asyncio.Semaphore) -> RefuterResult:
        adv_query = f"evidence contradicting or disproving: {claim.claim_text}"
        
        counter_passages = index.retrieve(adv_query, k=self.k)
        retrieved_docs_text = "\n".join([f"[{p.doc_id}] {p.text}" for p in counter_passages])
        valid_doc_ids = {p.doc_id for p in counter_passages}
        
        system_content = self.system_prompt_template.replace("{claim_text}", claim.claim_text).replace("{retrieved_documents}", retrieved_docs_text)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": getattr(self.config.llm, "model", "llama-3.3-70b-versatile"),
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Analyze the claim against the retrieved counter-evidence and output a valid JSON verdict."}
            ],
            "temperature": getattr(self.config.llm, "temperature", 0.1),
            "max_tokens": getattr(self.config.llm, "max_tokens", 2048),
            "response_format": {"type": "json_object"}
        }
        
        max_retries = getattr(self.config.refuter, "max_retries", 2)
        backoff = getattr(self.config.refuter, "retry_backoff_seconds", [1, 2])
        
        result_json = None
        
        async with semaphore:
            async with httpx.AsyncClient(timeout=60.0) as client:
                for attempt in range(max_retries + 1):
                    try:
                        response = await client.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers=headers,
                            json=payload
                        )
                        response.raise_for_status()
                        data = response.json()
                        content = data["choices"][0]["message"]["content"]
                        result_json = json.loads(content)
                        break
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code in [429, 503] and attempt < max_retries:
                            wait_time = backoff[attempt] if attempt < len(backoff) else backoff[-1]
                            await asyncio.sleep(wait_time)
                            continue
                        raise
                    except json.JSONDecodeError:
                        if attempt == 0:
                            payload["messages"][-1]["content"] += "\n\nCRITICAL: Output valid JSON only."
                            continue
                        raise
                    except Exception as e:
                        if attempt < max_retries:
                            wait_time = backoff[attempt] if attempt < len(backoff) else backoff[-1]
                            await asyncio.sleep(wait_time)
                            continue
                        raise
                        
        if result_json is None:
            raise Exception("Failed to get valid JSON from Refuter")
            
        verdict = result_json.get("verdict", "NOT_FOUND")
        if verdict not in ["CONTESTED", "INSUFFICIENT", "NOT_FOUND"]:
            verdict = "NOT_FOUND"
            
        json_passages = result_json.get("counter_passages", [])
        out_passages = []
        leakage_flag = False
        
        for p in json_passages:
            doc_id = p.get("doc_id", "")
            if doc_id and doc_id not in valid_doc_ids:
                leakage_flag = True
                
            out_passages.append(RetrievedPassage(
                doc_id=doc_id,
                source_title="Refuter Result",
                chunk_index=0,
                text=p.get("passage", ""),
                similarity_score=0.0
            ))
            
        return RefuterResult(
            claim_id=claim.claim_id,
            verdict=verdict,
            counter_passages=out_passages,
            query_used=result_json.get("query_used", adv_query),
            leakage_flag=leakage_flag
        )
        
    async def challenge_all(self, claims: List[Claim], index: VectorIndex) -> List[RefuterResult]:
        concurrency_cap = getattr(self.config.refuter, "concurrency_cap", 3)
        semaphore = asyncio.Semaphore(concurrency_cap)
        
        tasks = [self.challenge(c, index, semaphore) for c in claims]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        final_results = []
        for c, res in zip(claims, results):
            if isinstance(res, Exception):
                print(f"Exception during challenge for claim {c.claim_id}: {res}")
                final_results.append(RefuterResult(
                    claim_id=c.claim_id,
                    verdict="NOT_FOUND",
                    counter_passages=[],
                    query_used=f"evidence contradicting or disproving: {c.claim_text}",
                    leakage_flag=False
                ))
            else:
                final_results.append(res)
                
        return final_results

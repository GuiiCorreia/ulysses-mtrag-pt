"""
Cross-Encoder Reranker via DeepInfra API.

Implements neural reranking using Qwen3-Reranker models hosted on DeepInfra.
Cross-encoders jointly encode (query, document) pairs to compute relevance
scores — more accurate than bi-encoder retrieval but slower (O(n) calls).

Models available on DeepInfra (2025):
  - Qwen/Qwen3-Reranker-0.6B  ($0.010/1M tokens) — best cost/quality ratio
  - Qwen/Qwen3-Reranker-4B    ($0.025/1M tokens) — balanced
  - Qwen/Qwen3-Reranker-8B    ($0.050/1M tokens) — highest quality

Reference: Qwen Team (2025). Qwen3 Technical Report.

Usage:
    reranker = Qwen3Reranker(model="Qwen/Qwen3-Reranker-0.6B")
    reranked = reranker.rerank(query="cavalos de raça", results=bm25_results, top_k=5)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from mestrado.data.schema import Bill
from mestrado.retrieval.bm25 import RetrievalResult

logger = logging.getLogger(__name__)

# Qwen3 Reranker uses a specific prompt format
RERANKER_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)

QWEN3_RERANKER_PROMPT = """<Instruct>: {instruction}
<Query>: {query}
<Document>: {document}"""


@dataclass
class RerankedResult:
    """Reranked retrieval result with reranker score."""
    bill: Bill
    original_rank: int
    reranker_rank: int
    reranker_score: float
    retrieval_score: float


class Qwen3Reranker:
    """
    Neural cross-encoder reranker using Qwen3-Reranker via DeepInfra.

    Takes BM25/Dense retrieval results and re-scores them using a
    cross-encoder model that jointly encodes the query-document pair.

    This implements a two-stage retrieval pipeline:
      Stage 1: BM25/Dense → top-K candidates (fast, lexical/semantic)
      Stage 2: Cross-encoder → top-k final results (slow, accurate)

    The two-stage approach is the standard RAG retrieval pattern per:
    - Nogueira & Cho (2019). Passage Re-ranking with BERT.
    - Ma et al. (2023). Fine-Tuning LLaMA for Multi-Stage Text Retrieval.
    """

    # DeepInfra inference endpoint (not OpenAI-compatible)
    DEEPINFRA_INFERENCE_URL = "https://api.deepinfra.com/v1/inference"

    def __init__(
        self,
        model: str = "Qwen/Qwen3-Reranker-0.6B",
        api_key: str | None = None,
        instruction: str = RERANKER_INSTRUCTION,
        request_delay: float = 0.3,
    ) -> None:
        import os
        from dotenv import load_dotenv
        load_dotenv()
        self.model = model
        self.instruction = instruction
        self.request_delay = request_delay
        self._api_key = api_key or os.getenv("DEEPINFRA_API_KEY")
        if not self._api_key:
            raise ValueError("DEEPINFRA_API_KEY not set")
        logger.info(f"Qwen3Reranker initialized: {model}")

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """
        Rerank retrieval results using the cross-encoder.

        Args:
            query: The original user query.
            results: Candidate documents from BM25/Dense retrieval.
            top_k: Number of top documents to return after reranking.

        Returns:
            List of RerankedResult sorted by reranker score (descending).
        """
        if not results:
            return []

        scored = []
        for result in results:
            score = self._score_pair(query, result.bill.txt_ementa or result.bill.name)
            scored.append((result, score))
            time.sleep(self.request_delay)

        # Sort by reranker score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        reranked = []
        for new_rank, (result, score) in enumerate(scored[:top_k], start=1):
            reranked.append(RerankedResult(
                bill=result.bill,
                original_rank=result.rank,
                reranker_rank=new_rank,
                reranker_score=score,
                retrieval_score=result.score,
            ))

        return reranked

    def _score_pair(self, query: str, document: str) -> float:
        """
        Score a single (query, document) pair.
        Uses DeepInfra's inference endpoint with the Qwen3 reranker format.
        """
        import urllib.request
        import json

        prompt = QWEN3_RERANKER_PROMPT.format(
            instruction=self.instruction,
            query=query,
            document=document[:1500],  # Truncate long documents
        )

        payload = json.dumps({
            "input": prompt,
            "normalize": True,  # Returns score in [0, 1]
        }).encode()

        url = f"{self.DEEPINFRA_INFERENCE_URL}/{self.model}"
        req = urllib.request.Request(
            url,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
                # DeepInfra reranker returns scores list
                scores = data.get("scores", data.get("logits", [0.5]))
                return float(scores[0]) if scores else 0.5
        except Exception as e:
            logger.warning(f"Reranker score failed: {e}")
            return 0.0

    @classmethod
    def available_models(cls) -> dict[str, float]:
        """Return available models with their price per 1M tokens."""
        return {
            "Qwen/Qwen3-Reranker-0.6B": 0.010,
            "Qwen/Qwen3-Reranker-4B":   0.025,
            "Qwen/Qwen3-Reranker-8B":   0.050,
            "nvidia/llama-nemotron-rerank-vl-1b-v2": 0.010,
        }

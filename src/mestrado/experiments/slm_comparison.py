"""
SLM Comparison Experiment — multi-model RAG evaluation.

Compares 7 SLMs (2025 state-of-the-art) against an LLM oracle for RAG generation
quality in a given dataset. Produces results for H5 analysis.

Pipeline (same as paper reference Vitório et al. 2024):
  query → BM25 top-12 bills → ementa context → model generates answer
  oracle (LLM 70B) runs first → SLMs compared to oracle via ROUGE-L + BERTScore
  Faithfulness evaluated independently per model via LLM-as-judge.

Models tested (see mestrado/slm_comparison/TODOS_MODELOS_DEEPINFRA.md for full analysis):
  - google/gemma-3-4b-it            (Google 2025, 4B) ✅ Validated
  - meta-llama/Meta-Llama-3.1-8B-Instruct (Meta 2024, 8B) ✅ Validated
  - meta-llama/Llama-3.2-3B-Instruct (Meta 2024, 3B) ✅ Validated
  - meta-llama/Llama-3.2-1B-Instruct (Meta 2024, 1B) ✅ Validated
  - Qwen/Qwen2.5-1.5B-Instruct      (Qwen 2024, 1.5B) ✅ Validated
  - Qwen/Qwen2.5-3B-Instruct        (Qwen 2024, 3B) ✅ Validated
  - Qwen/Qwen2.5-7B-Instruct        (Qwen 2024, 7B) ✅ Validated
  Oracle: meta-llama/Llama-3.3-70B-Instruct (Meta 2024, reference quality)

CRITICAL BUG DISCOVERED (2026-04-07 diagnostic):
  ALL Qwen3.5 models (0.8B, 2B, 4B, 9B) have systematic bug on DeepInfra API.
  Diagnostic results: tokens generated (50-1024) but message.content = ''.
  Bug confirmed across ALL scenarios (simple prompt, RAG, all max_tokens values).
  Root cause: DeepInfra API bug for Qwen3.5 family, not implementation issue.
  Solution: Use Qwen2.5 models (1.5B, 3B, 7B) which work perfectly.

Full analysis of all SLM models on DeepInfra (≤10B parameters):
  See mestrado/slm_comparison/TODOS_MODELOS_DEEPINFRA.md

References:
  - Belcak & Wattenhofer (2025). Small Language Models Are the Future of
    Agentic AI. arXiv:2506.02153.
  - Google (2025). Gemma 3 Technical Report. arXiv:2503.19786.
  - Dubey et al. (2024). The Llama 3 Herd of Models. arXiv:2407.21783.
  - Meta AI (2024). Llama 3.2 Model Card.
  - Qwen Team (2024). Qwen2.5 Technical Report. arXiv:2309.16609.
  - Sun et al. (2023). Is ChatGPT Good at Search? EMNLP 2023. [reranker pattern]
  - Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench. NeurIPS 2023.
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from mestrado.data.schema import Bill, Query
from mestrado.generation.rag import DeepInfraRAG, RAGResult
from mestrado.metrics.generation_metrics import evaluate_generation, GenerationMetrics
from mestrado.reranking.llm_reranker import DeepInfraReranker, RerankerResult
from mestrado.retrieval.bm25 import BM25Retriever, RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class ModelResult:
    """Results for a single model on all queries."""
    model: str
    role: str  # "oracle", "slm", or "slm_reranker"
    responses: list[str] = field(default_factory=list)
    faithfulness_scores: list[float] = field(default_factory=list)
    tokens_in_list: list[int] = field(default_factory=list)
    tokens_out_list: list[int] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    generation_metrics: Optional[GenerationMetrics] = None

    @property
    def avg_tokens_in(self) -> float:
        return sum(self.tokens_in_list) / max(1, len(self.tokens_in_list))

    @property
    def avg_tokens_out(self) -> float:
        return sum(self.tokens_out_list) / max(1, len(self.tokens_out_list))

    @property
    def avg_latency_s(self) -> float:
        return sum(self.latencies) / max(1, len(self.latencies))

    @property
    def avg_faithfulness(self) -> float:
        valid = [s for s in self.faithfulness_scores if s is not None]
        return sum(valid) / max(1, len(valid))

    def to_summary_dict(self) -> dict:
        d = {
            "model": self.model,
            "role": self.role,
            "n_queries": len(self.responses),
            "avg_latency_s": round(self.avg_latency_s, 2),
            "avg_tokens_in": round(self.avg_tokens_in),
            "avg_tokens_out": round(self.avg_tokens_out),
            "avg_faithfulness": round(self.avg_faithfulness, 4),
        }
        if self.generation_metrics:
            d.update(self.generation_metrics.to_dict())
        return d


@dataclass
class SLMComparisonResults:
    """Full results of the SLM comparison experiment."""
    dataset_name: str
    n_queries: int
    oracle_model: str
    slm_models: list[str]
    model_results: dict[str, ModelResult] = field(default_factory=dict)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset_name,
            "n_queries": self.n_queries,
            "oracle_model": self.oracle_model,
            "slm_models": self.slm_models,
            "timestamp": self.timestamp,
            "results": {
                model: result.to_summary_dict()
                for model, result in self.model_results.items()
            },
        }


class SLMComparisonRunner:
    """
    Orchestrates multi-model RAG comparison experiments.

    Workflow:
      1. Load BM25 index (retrieval identical across all models)
      2. Run oracle LLM → save oracle_outputs as reference
      3. Run each SLM → compare to oracle via ROUGE-L + BERTScore
      4. Evaluate faithfulness per model via LLM-as-judge
      5. Save full results + summary table

    Usage:
        runner = SLMComparisonRunner(bills, queries, dataset_name="ulysses")
        runner.setup()
        results = runner.run(oracle_only=False)
        runner.save(results, Path("results/"))
    """

    def __init__(
        self,
        bills_by_name: dict[str, Bill],
        queries: list[Query],
        dataset_name: str = "ulysses",
        n_queries: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        from mestrado.config import slm_comparison as cfg
        self.bills_by_name = bills_by_name
        self.dataset_name = dataset_name
        self.seed = seed

        # Sample queries reproducibly
        all_queries = [q for q in queries if q.has_relevant]
        if n_queries and n_queries < len(all_queries):
            rng = random.Random(seed)
            self.queries = rng.sample(all_queries, n_queries)
            logger.info(
                f"Sampled {n_queries} queries from {len(all_queries)} "
                f"(seed={seed}, dataset={dataset_name})"
            )
        else:
            self.queries = all_queries
            logger.info(f"Using all {len(self.queries)} queries (dataset={dataset_name})")

        self.oracle_model = cfg.oracle_model
        self.slm_models = cfg.slm_models
        self._bm25: Optional[BM25Retriever] = None

    def setup(self) -> "SLMComparisonRunner":
        """Build BM25 index — used identically for all models."""
        logger.info(f"Building BM25 index ({len(self.bills_by_name):,} documents)...")
        self._bm25 = BM25Retriever(self.bills_by_name)
        self._bm25.build_index()
        return self

    def run(
        self,
        oracle_only: bool = False,
        top_k_retrieval: int = 50,
        top_n_context: int = 5,
    ) -> SLMComparisonResults:
        """
        Run the full SLM comparison experiment.

        Args:
            oracle_only: If True, only run the oracle LLM (E6). Useful for
                         validating prompts and estimating cost before running SLMs.
            top_k_retrieval: BM25 candidates before context selection.
            top_n_context: Bills included in RAG context (default 5).
        """
        assert self._bm25 is not None, "Call setup() first."

        results = SLMComparisonResults(
            dataset_name=self.dataset_name,
            n_queries=len(self.queries),
            oracle_model=self.oracle_model,
            slm_models=[] if oracle_only else self.slm_models,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )

        # Step 1: Retrieve top-k bills for all queries (done once, shared)
        logger.info("Retrieving BM25 top-k for all queries...")
        retrieved: dict[str, list[RetrievalResult]] = {}
        for q in tqdm(self.queries, desc="BM25 retrieval"):
            retrieved[q.query_id] = self._bm25.retrieve(q.text, top_k=top_k_retrieval)

        # Convert to RerankerResult for RAG interface
        def to_reranker_results(results_list: list[RetrievalResult]) -> list[RerankerResult]:
            return [RerankerResult(bill=r.bill, rank=r.rank) for r in results_list]

        # Step 2: Oracle LLM (E6 reference)
        logger.info(f"Running oracle LLM: {self.oracle_model}")
        oracle_result = self._run_model(
            model=self.oracle_model,
            role="oracle",
            retrieved=retrieved,
            to_reranker_fn=to_reranker_results,
            top_n=top_n_context,
        )
        results.model_results[self.oracle_model] = oracle_result

        if oracle_only:
            logger.info("oracle_only=True — stopping after oracle run.")
            return results

        # Step 3: Each SLM compared to oracle
        oracle_responses = oracle_result.responses
        for slm_model in self.slm_models:
            logger.info(f"Running SLM: {slm_model}")
            slm_result = self._run_model(
                model=slm_model,
                role="slm",
                retrieved=retrieved,
                to_reranker_fn=to_reranker_results,
                top_n=top_n_context,
            )
            # Compute generation metrics vs. oracle outputs (H5 evaluation)
            if oracle_responses and slm_result.responses:
                slm_result.generation_metrics = evaluate_generation(
                    predictions=slm_result.responses,
                    references=oracle_responses,
                    lang="pt",
                )
                logger.info(
                    f"  {slm_model}: ROUGE-L={slm_result.generation_metrics.rouge_l_mean:.4f}, "
                    f"BERTScore={slm_result.generation_metrics.bertscore_f1_mean:.4f}, "
                    f"Faithfulness={slm_result.avg_faithfulness:.4f}"
                )
            results.model_results[slm_model] = slm_result

        return results

    def _run_model(
        self,
        model: str,
        role: str,
        retrieved: dict[str, list[RetrievalResult]],
        to_reranker_fn,
        top_n: int,
    ) -> ModelResult:
        """Run RAG generation for one model on all queries."""
        rag = DeepInfraRAG(model=model, context_mode="ementa")
        model_result = ModelResult(model=model, role=role)

        for q in tqdm(self.queries, desc=f"  {model.split('/')[-1]}", leave=False):
            reranked = to_reranker_fn(retrieved.get(q.query_id, []))
            gen: RAGResult = rag.generate(q.text, reranked, top_bills=top_n)

            model_result.responses.append(gen.response)
            model_result.tokens_in_list.append(gen.tokens_context)
            model_result.tokens_out_list.append(gen.tokens_generated)
            model_result.latencies.append(gen.latency_s)

            # Faithfulness evaluation (LLM-as-judge, same model)
            faith = rag.evaluate_faithfulness(gen)
            score = faith.get("score")
            model_result.faithfulness_scores.append(
                float(score) if score is not None else 0.0
            )

        return model_result

    @staticmethod
    def save(results: SLMComparisonResults, results_dir: Path) -> Path:
        """Save full results JSON and print summary table."""
        results_dir.mkdir(exist_ok=True)
        ts = results.timestamp.replace(":", "-").replace("T", "_")
        out_path = results_dir / f"slm_comparison_{results.dataset_name}_{ts}.json"

        # Full output (all responses + per-query scores)
        full_data = results.to_dict()
        full_data["per_query"] = {
            model: {
                "responses": mr.responses,
                "faithfulness": mr.faithfulness_scores,
                "latencies": mr.latencies,
            }
            for model, mr in results.model_results.items()
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(full_data, f, ensure_ascii=False, indent=2)

        logger.info(f"Results saved to {out_path}")

        # Print summary table
        print(f"\n{'='*80}")
        print(f"SLM Comparison Results — {results.dataset_name.upper()} ({results.n_queries} queries)")
        print(f"{'='*80}")
        header = f"{'Model':<45} {'ROUGE-L':>8} {'BERTScore':>10} {'Faith':>7} {'Lat(s)':>7}"
        print(header)
        print("-" * 80)
        for model, mr in results.model_results.items():
            rouge = mr.generation_metrics.rouge_l_mean if mr.generation_metrics else 0.0
            bert = mr.generation_metrics.bertscore_f1_mean if mr.generation_metrics else 0.0
            faith = mr.avg_faithfulness
            lat = mr.avg_latency_s
            label = "ORACLE" if mr.role == "oracle" else ""
            name = model.split("/")[-1][:40]
            print(f"{name:<45} {rouge:>8.4f} {bert:>10.4f} {faith:>7.4f} {lat:>7.1f} {label}")
        print("=" * 80)

        return out_path

"""
ExperimentRunner — orchestrates all 8 experiments for BillsRAG-BR benchmark.

Experiments:
  E1: BM25 baseline
  E2: Dense retrieval (sentence-transformers + FAISS)
  E3: Hybrid retrieval (BM25 + Dense + RRF)
  E4: Two-stage: Hybrid retrieval + LLM reranker (Ollama)
  E5: Query expansion + BM25
  E6: RAG generation (LLM via Ollama — full context)
  E7: RAG generation (SLM via Ollama — ementa context)
  E8: RAG generation (SLM + structured summaries — Exp 8)

Hypotheses tested:
  H1: Dense/Hybrid > BM25 (E2/E3 vs E1)
  H2: LLM Reranker improves Hybrid (E4 vs E3)
  H3: Query Expansion improves BM25 recall (E5 vs E1)
  H4: RAG generation quality LLM ≥ SLM×threshold (E6 vs E7)
  H5: Structured summaries reduce SLM degradation (E8 vs E7)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from mestrado.data.loader import DataLoader
from mestrado.data.schema import Bill, Query
from mestrado.metrics.ir_metrics import EvaluationSuite, SystemResult
from mestrado.retrieval.bm25 import BM25Retriever, RetrievalResult
from mestrado.retrieval.dense import DenseRetriever
from mestrado.retrieval.hybrid import HybridRetriever
from mestrado.reranking.llm_reranker import OllamaReranker
from mestrado.generation.rag import OllamaRAG
from mestrado.utils.context_compactor import ContextCompactor

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a specific experiment run."""
    name: str
    enabled: bool = True
    top_k_retrieval: int = 50
    rerank_top_n: int = 12
    extra: dict = field(default_factory=dict)


class ExperimentRunner:
    """
    Orchestrates all BillsRAG-BR experiments with consistent evaluation.

    Usage:
        runner = ExperimentRunner()
        runner.setup()  # loads data, builds indexes
        results = runner.run_all()
        runner.save_and_report(results)
    """

    def __init__(
        self,
        sample_size: int | None = None,
        run_generation: bool = False,
        results_dir: Path | None = None,
    ) -> None:
        from mestrado.config import experiment as exp_cfg, output as out_cfg
        self.sample_size = sample_size or exp_cfg.sample_size
        self.run_generation = run_generation
        self.results_dir = results_dir or out_cfg.results_dir
        self.top_k = exp_cfg.top_k_retrieval
        self.top_n = exp_cfg.rerank_top_n

        # Populated by setup()
        self.bills_by_name: dict[str, Bill] = {}
        self.queries: list[Query] = []
        self.suite: EvaluationSuite | None = None

        # Retrievers (populated on demand)
        self._bm25: BM25Retriever | None = None
        self._dense: DenseRetriever | None = None
        self._hybrid: HybridRetriever | None = None
        self._reranker: OllamaReranker | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, full_corpus: bool = True) -> "ExperimentRunner":
        """Load data and build retrieval indexes."""
        logger.info("=== Setup: Loading datasets ===")
        loader = DataLoader()
        self.bills_by_name, self.queries = loader.load_all(
            sample_size=None if full_corpus else self.sample_size
        )
        self.suite = EvaluationSuite(self.queries)

        logger.info("=== Setup: Building BM25 index ===")
        self._bm25 = BM25Retriever(self.bills_by_name)
        self._bm25.build_index()

        return self

    def setup_dense(self) -> "ExperimentRunner":
        """Build dense index (separate — slow, requires sentence-transformers)."""
        logger.info("=== Setup: Building Dense index ===")
        cache = self.results_dir / "dense_embeddings.pkl"
        self._dense = DenseRetriever(self.bills_by_name)
        self._dense.build_index(cache_path=cache)
        self._hybrid = HybridRetriever(bm25=self._bm25, dense=self._dense)
        return self

    # ------------------------------------------------------------------
    # Individual experiments
    # ------------------------------------------------------------------

    def run_e1_bm25(self) -> SystemResult:
        """Experiment 1: BM25 baseline."""
        logger.info("=== E1: BM25 Baseline ===")
        assert self._bm25 is not None, "Call setup() first."
        t0 = time.time()

        def retrieve(q: Query) -> list[RetrievalResult]:
            return self._bm25.retrieve(q.text, top_k=self.top_n)

        result = self.suite.evaluate("E1_BM25", retrieve)
        result.extra["latency_total_s"] = round(time.time() - t0, 1)
        result.extra["cost_usd"] = 0.0
        return result

    def run_e2_dense(self) -> SystemResult:
        """Experiment 2: Dense retrieval."""
        logger.info("=== E2: Dense Retrieval ===")
        assert self._dense is not None, "Call setup_dense() first."
        t0 = time.time()

        def retrieve(q: Query) -> list[RetrievalResult]:
            return self._dense.retrieve(q.text, top_k=self.top_n)

        result = self.suite.evaluate("E2_Dense", retrieve)
        result.extra["latency_total_s"] = round(time.time() - t0, 1)
        result.extra["cost_usd"] = 0.0
        return result

    def run_e3_hybrid(self) -> SystemResult:
        """Experiment 3: Hybrid retrieval (BM25 + Dense + RRF)."""
        logger.info("=== E3: Hybrid RRF ===")
        assert self._hybrid is not None, "Call setup_dense() first."
        t0 = time.time()

        def retrieve(q: Query) -> list[RetrievalResult]:
            return self._hybrid.retrieve(q.text, top_k=self.top_n)

        result = self.suite.evaluate("E3_Hybrid_RRF", retrieve)
        result.extra["latency_total_s"] = round(time.time() - t0, 1)
        result.extra["cost_usd"] = 0.0
        return result

    def run_e4_llm_reranker(
        self,
        stage1_retriever: str = "hybrid",
    ) -> SystemResult:
        """
        Experiment 4: Two-stage — Stage 1 retrieval + Stage 2 LLM reranker (Ollama).
        Direct adaptation of findRelevantMemories.ts pattern.
        """
        logger.info(f"=== E4: LLM Reranker (stage1={stage1_retriever}, model={self._get_reranker().model}) ===")
        reranker = self._get_reranker()

        # Use sample for LLM experiments (speed)
        sample_queries = self.queries[:self.sample_size]
        sample_suite = EvaluationSuite(sample_queries)

        t0 = time.time()

        def retrieve(q: Query) -> list[RetrievalResult]:
            # Stage 1
            if stage1_retriever == "hybrid" and self._hybrid:
                candidates = self._hybrid.retrieve(q.text, top_k=self.top_k)
            else:
                candidates = self._bm25.retrieve(q.text, top_k=self.top_k)

            # Stage 2: LLM reranks
            reranked = reranker.rerank(q.text, candidates, top_n=self.top_n)
            return [RetrievalResult(bill=r.bill, score=1.0 / r.rank, rank=r.rank)
                    for r in reranked]

        result = sample_suite.evaluate(
            f"E4_LLM_Reranker_{reranker.model}",
            retrieve,
        )
        total_time = time.time() - t0
        result.extra.update({
            "latency_total_s": round(total_time, 1),
            "avg_latency_s": round(reranker.stats.avg_latency_s, 2),
            "total_tokens_in": reranker.stats.total_tokens_in,
            "total_tokens_out": reranker.stats.total_tokens_out,
            "hallucinations_filtered": reranker.stats.hallucinations_filtered,
            "cost_usd": 0.0,  # Ollama local
            "model": reranker.model,
            "n_queries_sampled": len(sample_queries),
        })
        return result

    def run_e5_query_expansion(self) -> SystemResult:
        """
        Experiment 5: Query expansion — translates informal query to legislative language.
        Tests H3: Query Expansion improves recall for semantically mismatched queries.
        """
        logger.info("=== E5: Query Expansion + BM25 ===")
        reranker = self._get_reranker()
        sample_queries = self.queries[:self.sample_size]
        sample_suite = EvaluationSuite(sample_queries)
        expansion_cache: dict[str, str] = {}
        t0 = time.time()

        def retrieve(q: Query) -> list[RetrievalResult]:
            if q.query_id not in expansion_cache:
                expanded = reranker.expand_query(q.text)
                expansion_cache[q.query_id] = expanded.get("query_expanded", q.text)
            expanded_text = expansion_cache[q.query_id]
            return self._bm25.retrieve(expanded_text, top_k=self.top_n)

        result = sample_suite.evaluate("E5_QueryExpansion_BM25", retrieve)
        result.extra["latency_total_s"] = round(time.time() - t0, 1)
        result.extra["cost_usd"] = 0.0
        result.extra["expansions_generated"] = len(expansion_cache)

        # Save expansions for qualitative analysis
        exp_path = self.results_dir / "query_expansions.json"
        exp_path.parent.mkdir(exist_ok=True)
        with open(exp_path, "w", encoding="utf-8") as f:
            json.dump(expansion_cache, f, ensure_ascii=False, indent=2)

        return result

    def run_e6_rag_llm(self, retriever: str = "hybrid") -> SystemResult:
        """
        Experiment 6: RAG generation with LLM (Ollama — full context).
        Evaluates generation quality using ROUGE-L and faithfulness.
        Note: nDCG metrics come from retrieval stage; generation evaluated separately.
        """
        logger.info("=== E6: RAG Generation (LLM, full context) ===")
        rag = OllamaRAG(context_mode="full", max_context_tokens=8192)
        sample_queries = self.queries[:min(50, self.sample_size)]  # 50 for generation
        sample_suite = EvaluationSuite(sample_queries)
        generation_results = []
        t0 = time.time()

        def retrieve(q: Query) -> list[RetrievalResult]:
            if self._hybrid:
                candidates = self._hybrid.retrieve(q.text, top_k=self.top_k)
            else:
                candidates = self._bm25.retrieve(q.text, top_k=self.top_k)
            reranked = [RetrievalResult(bill=r.bill, score=1.0 / (i+1), rank=i+1)
                        for i, r in enumerate(candidates[:self.top_n])]
            # Generate response
            from mestrado.reranking.llm_reranker import RerankerResult
            reranked_results = [RerankerResult(bill=r.bill, rank=r.rank) for r in reranked]
            gen = rag.generate(q.text, reranked_results, top_bills=3)
            generation_results.append({
                "query_id": q.query_id,
                "query": q.text,
                "response": gen.response,
                "bills_cited": [b.name for b in gen.bills_used],
                "tokens_generated": gen.tokens_generated,
                "latency_s": gen.latency_s,
                "truncated": gen.truncated,
            })
            return reranked

        result = sample_suite.evaluate("E6_RAG_LLM_Full", retrieve)
        result.extra["latency_total_s"] = round(time.time() - t0, 1)
        result.extra["cost_usd"] = 0.0
        result.extra["model"] = rag.model

        gen_path = self.results_dir / "e6_generation_results.json"
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(generation_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Generation results saved to {gen_path}")

        return result

    def run_e7_rag_slm(self) -> SystemResult:
        """
        Experiment 7: RAG generation with SLM (Ollama — ementa context only).
        Tests H4: SLM quality relative to LLM (Exp 6).
        Uses ementa-only context to fit within SLM window.
        """
        logger.info("=== E7: RAG Generation (SLM, ementa context) ===")
        rag = OllamaRAG(context_mode="ementa", max_context_tokens=4096)
        sample_queries = self.queries[:min(50, self.sample_size)]
        sample_suite = EvaluationSuite(sample_queries)
        generation_results = []
        t0 = time.time()

        def retrieve(q: Query) -> list[RetrievalResult]:
            candidates = self._bm25.retrieve(q.text, top_k=self.top_k)
            reranked = candidates[:self.top_n]
            from mestrado.reranking.llm_reranker import RerankerResult
            reranked_results = [RerankerResult(bill=r.bill, rank=i+1) for i, r in enumerate(reranked)]
            gen = rag.generate(q.text, reranked_results, top_bills=5)
            generation_results.append({
                "query_id": q.query_id,
                "query": q.text,
                "response": gen.response,
                "tokens_generated": gen.tokens_generated,
                "latency_s": gen.latency_s,
                "truncated": gen.truncated,
            })
            return reranked

        result = sample_suite.evaluate("E7_RAG_SLM_Ementa", retrieve)
        result.extra["latency_total_s"] = round(time.time() - t0, 1)
        result.extra["cost_usd"] = 0.0
        result.extra["model"] = rag.model

        gen_path = self.results_dir / "e7_generation_results.json"
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(generation_results, f, ensure_ascii=False, indent=2)

        return result

    def run_e8_rag_slm_with_summaries(self) -> SystemResult:
        """
        Experiment 8: RAG generation with SLM + structured summaries (Exp 8).
        Tests H5: Structured summarization reduces SLM quality degradation.
        Uses ContextCompactor (adapted from compact/prompt.ts) to pre-summarize bills.
        """
        logger.info("=== E8: RAG Generation (SLM + Structured Summaries) ===")
        compactor = ContextCompactor()
        cache_path = self.results_dir / "bill_summaries.pkl"
        compactor.load_cache(cache_path)

        sample_queries = self.queries[:min(50, self.sample_size)]
        sample_suite = EvaluationSuite(sample_queries)

        # Identify which bills appear in sample queries and pre-summarize
        relevant_bill_names = set()
        for q in sample_queries:
            candidates = self._bm25.retrieve(q.text, top_k=self.top_k)
            for r in candidates[:self.top_n]:
                relevant_bill_names.add(r.bill.name)

        bills_to_summarize = [
            b for n, b in self.bills_by_name.items()
            if n in relevant_bill_names
        ]
        logger.info(f"Pre-summarizing {len(bills_to_summarize)} bills...")
        summaries = compactor.summarize_batch(bills_to_summarize)
        compactor.save_cache(cache_path)

        rag = OllamaRAG(context_mode="summary", max_context_tokens=4096)
        generation_results = []
        t0 = time.time()

        def retrieve(q: Query) -> list[RetrievalResult]:
            candidates = self._bm25.retrieve(q.text, top_k=self.top_k)
            reranked = candidates[:self.top_n]

            # Override bill text with structured summary for generation
            from mestrado.reranking.llm_reranker import RerankerResult
            reranked_results = []
            for i, r in enumerate(reranked):
                summary = summaries.get(r.bill.name)
                bill = r.bill
                if summary:
                    # Patch bill with summary text for context (non-destructive)
                    import copy
                    patched = copy.replace(bill, txt_ementa=summary.to_retrieval_text())
                    reranked_results.append(RerankerResult(bill=patched, rank=i+1))
                else:
                    reranked_results.append(RerankerResult(bill=bill, rank=i+1))

            gen = rag.generate(q.text, reranked_results, top_bills=5)
            generation_results.append({
                "query_id": q.query_id,
                "query": q.text,
                "response": gen.response,
                "tokens_generated": gen.tokens_generated,
                "latency_s": gen.latency_s,
                "summaries_used": len([r for r in reranked_results
                                       if r.bill.name in summaries]),
            })
            return reranked

        result = sample_suite.evaluate("E8_RAG_SLM_Summaries", retrieve)
        result.extra["latency_total_s"] = round(time.time() - t0, 1)
        result.extra["cost_usd"] = 0.0
        result.extra["model"] = rag.model
        result.extra["bills_summarized"] = len(summaries)

        gen_path = self.results_dir / "e8_generation_results.json"
        with open(gen_path, "w", encoding="utf-8") as f:
            json.dump(generation_results, f, ensure_ascii=False, indent=2)

        return result

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def run_all(
        self,
        experiments: list[str] | None = None,
    ) -> dict[str, SystemResult]:
        """
        Run all (or selected) experiments and return results dict.

        experiments: list of experiment IDs to run, e.g. ["e1", "e3", "e4"]
                     None = run all retrieval experiments (E1–E5)
        """
        all_ids = experiments or ["e1", "e2", "e3", "e4", "e5"]
        results: dict[str, SystemResult] = {}

        if "e1" in all_ids:
            results["e1"] = self.run_e1_bm25()

        if "e2" in all_ids or "e3" in all_ids:
            self.setup_dense()

        if "e2" in all_ids:
            results["e2"] = self.run_e2_dense()

        if "e3" in all_ids:
            results["e3"] = self.run_e3_hybrid()

        if "e4" in all_ids:
            results["e4"] = self.run_e4_llm_reranker(
                stage1_retriever="hybrid" if self._hybrid else "bm25"
            )

        if "e5" in all_ids:
            results["e5"] = self.run_e5_query_expansion()

        if self.run_generation:
            if "e6" in all_ids:
                results["e6"] = self.run_e6_rag_llm()
            if "e7" in all_ids:
                results["e7"] = self.run_e7_rag_slm()
            if "e8" in all_ids:
                results["e8"] = self.run_e8_rag_slm_with_summaries()

        return results

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def run_significance_tests(self, results: dict[str, SystemResult]) -> list[dict]:
        """
        Run all pairwise significance tests (Wilcoxon + Bonferroni).
        Tests H1–H5 as declared in ideias/09_contribuicoes_originais.md.
        """
        comparisons = []
        baseline = results.get("e1")
        if baseline is None:
            return []

        # Bonferroni: n_comparisons = 4 (H1: E2/E3 vs E1, H2: E4 vs E3, etc.)
        n_comp = len(results) - 1
        alpha_bonferroni = 0.05 / max(n_comp, 1)

        pairs = [
            ("e1", "e2", "H1: Dense > BM25"),
            ("e1", "e3", "H1: Hybrid > BM25"),
            ("e3", "e4", "H2: Reranker > Hybrid"),
            ("e1", "e5", "H3: QueryExpansion > BM25"),
        ]

        for a_key, b_key, label in pairs:
            if a_key in results and b_key in results:
                test = self.suite.significance_test(
                    results[a_key], results[b_key],
                    metric="ndcg10",
                    alpha=alpha_bonferroni,
                )
                test["hypothesis"] = label
                test["alpha_bonferroni"] = round(alpha_bonferroni, 4)
                comparisons.append(test)

        return comparisons

    def save_and_report(self, results: dict[str, SystemResult]) -> Path:
        """Save all results and print comparison table."""
        systems = list(results.values())
        self.suite.print_results(systems)

        # Statistical tests
        sig_tests = self.run_significance_tests(results)
        if sig_tests:
            logger.info("\n=== Significance Tests (Wilcoxon + Bonferroni) ===")
            for test in sig_tests:
                sig_marker = "✓ SIGNIFICANT" if test["significant"] else "✗ not significant"
                logger.info(
                    f"{test['hypothesis']}: "
                    f"Δ={test['delta']:+.4f} ({test['delta_pct']:+.1f}%), "
                    f"p={test['p_value']:.4f} {sig_marker}"
                )

        # Save JSON
        out_path = self.results_dir / "all_results.json"
        self.suite.save_results(systems, out_path)

        sig_path = self.results_dir / "significance_tests.json"
        with open(sig_path, "w") as f:
            json.dump(sig_tests, f, indent=2)

        logger.info(f"\nResults saved to {self.results_dir}/")
        return out_path

    def _get_reranker(self) -> OllamaReranker:
        if self._reranker is None:
            self._reranker = OllamaReranker()
        return self._reranker

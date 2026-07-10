"""
Cross-Domain Experiment — SLM comparison on Ulysses (legislative) + MIRACL PT (Wikipedia).

Runs the same SLM comparison pipeline on two datasets and produces a comparative
analysis quantifying whether SLMs suffer more from domain specialization than LLMs.

Research question:
  "Is the domain gap (legislative vs. general PT-BR) larger for SLMs than for LLMs?"
  If yes: supports the argument that SLMs need domain fine-tuning for specialized RAG.
  If no: supports Belcak & Wattenhofer (2025) claim that SLMs generalize well.

Datasets:
  - Ulysses-RFCorpus: legislative, specialized, formal PT-BR
    (Vitório et al., LRE 2024, DOI: 10.1007/s10579-024-09767-3)
  - MIRACL Portuguese: Wikipedia, general domain, natural PT-BR
    (Zhang et al., 2022, arXiv:2209.05299)

Usage:
    from mestrado.experiments.cross_domain import CrossDomainRunner
    runner = CrossDomainRunner(n_queries=100)
    analysis = runner.run()
    runner.save(analysis, Path("results/"))
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from mestrado.experiments.slm_comparison import SLMComparisonResults, SLMComparisonRunner
from mestrado.config import slm_comparison as cfg

logger = logging.getLogger(__name__)


@dataclass
class CrossDomainAnalysis:
    """
    Comparative analysis of SLM performance across two domains.
    The key metric is the "domain gap ratio" per model:
      domain_gap = (metric_miracl - metric_ulysses) / metric_miracl
    A larger positive value means the model degrades more in the specialized domain.
    """
    ulysses_results: Optional[SLMComparisonResults] = None
    miracl_results: Optional[SLMComparisonResults] = None
    timestamp: str = ""

    def domain_gap_table(self) -> list[dict]:
        """
        Compute domain gap per model for ROUGE-L and BERTScore.
        Returns list of rows for tabular display.
        """
        if not self.ulysses_results or not self.miracl_results:
            return []

        rows = []
        all_models = list(self.ulysses_results.model_results.keys())

        for model in all_models:
            ul_res = self.ulysses_results.model_results.get(model)
            mi_res = self.miracl_results.model_results.get(model)
            if not ul_res or not mi_res:
                continue

            ul_rouge = ul_res.generation_metrics.rouge_l_mean if ul_res.generation_metrics else 0.0
            mi_rouge = mi_res.generation_metrics.rouge_l_mean if mi_res.generation_metrics else 0.0
            ul_bert = ul_res.generation_metrics.bertscore_f1_mean if ul_res.generation_metrics else 0.0
            mi_bert = mi_res.generation_metrics.bertscore_f1_mean if mi_res.generation_metrics else 0.0

            rouge_gap = (mi_rouge - ul_rouge) / max(mi_rouge, 1e-6)
            bert_gap = (mi_bert - ul_bert) / max(mi_bert, 1e-6)

            rows.append({
                "model": model.split("/")[-1],
                "role": ul_res.role,
                "ulysses_rouge_l": round(ul_rouge, 4),
                "miracl_rouge_l": round(mi_rouge, 4),
                "domain_gap_rouge": round(rouge_gap, 4),
                "ulysses_bertscore": round(ul_bert, 4),
                "miracl_bertscore": round(mi_bert, 4),
                "domain_gap_bert": round(bert_gap, 4),
                "ulysses_faithfulness": round(ul_res.avg_faithfulness, 4),
                "miracl_faithfulness": round(mi_res.avg_faithfulness, 4),
            })

        return rows

    def print_comparison_table(self) -> None:
        """Print the cross-domain comparison table to stdout."""
        rows = self.domain_gap_table()
        if not rows:
            print("No cross-domain data to display.")
            return

        print(f"\n{'='*100}")
        print("CROSS-DOMAIN ANALYSIS — Ulysses (Legislative) vs. MIRACL (Wikipedia PT-BR)")
        print("Domain Gap = (MIRACL - Ulysses) / MIRACL  [positive = model degrades in specialized domain]")
        print("="*100)
        header = (
            f"{'Model':<35} {'UL ROUGE':>9} {'MI ROUGE':>9} {'Gap%':>7} "
            f"{'UL BERT':>8} {'MI BERT':>8} {'Gap%':>7} {'Role':>8}"
        )
        print(header)
        print("-"*100)
        for row in rows:
            gap_r = f"{row['domain_gap_rouge']*100:+.1f}%"
            gap_b = f"{row['domain_gap_bert']*100:+.1f}%"
            print(
                f"{row['model']:<35} "
                f"{row['ulysses_rouge_l']:>9.4f} {row['miracl_rouge_l']:>9.4f} {gap_r:>7} "
                f"{row['ulysses_bertscore']:>8.4f} {row['miracl_bertscore']:>8.4f} {gap_b:>7} "
                f"{row['role']:>8}"
            )
        print("="*100)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "domain_gap_table": self.domain_gap_table(),
            "ulysses": self.ulysses_results.to_dict() if self.ulysses_results else {},
            "miracl": self.miracl_results.to_dict() if self.miracl_results else {},
        }


class CrossDomainRunner:
    """
    Runs SLM comparison on both Ulysses-RFCorpus and MIRACL Portuguese.

    Usage:
        runner = CrossDomainRunner(n_queries=100)
        analysis = runner.run()
        runner.save(analysis, Path("results/"))

    For smoke test: CrossDomainRunner(n_queries=5).run(oracle_only=True)
    """

    def __init__(
        self,
        n_queries: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.n_queries = n_queries or cfg.n_queries
        self.seed = seed

    def run(self, oracle_only: bool = False) -> CrossDomainAnalysis:
        """Run both datasets and return comparative analysis."""
        analysis = CrossDomainAnalysis(timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"))

        # --- Ulysses-RFCorpus (legislative PT-BR) ---
        logger.info("=== DATASET 1/2: Ulysses-RFCorpus (Legislative PT-BR) ===")
        ulysses_bills, ulysses_queries = self._load_ulysses()
        ulysses_runner = SLMComparisonRunner(
            bills_by_name=ulysses_bills,
            queries=ulysses_queries,
            dataset_name="ulysses",
            n_queries=self.n_queries,
            seed=self.seed,
        ).setup()
        analysis.ulysses_results = ulysses_runner.run(oracle_only=oracle_only)

        # --- MIRACL Portuguese (Wikipedia general domain) ---
        logger.info("=== DATASET 2/2: MIRACL Portuguese (Wikipedia PT-BR) ===")
        miracl_bills, miracl_queries = self._load_miracl()
        miracl_runner = SLMComparisonRunner(
            bills_by_name=miracl_bills,
            queries=miracl_queries,
            dataset_name="miracl",
            n_queries=self.n_queries,
            seed=self.seed,
        ).setup()
        analysis.miracl_results = miracl_runner.run(oracle_only=oracle_only)

        analysis.print_comparison_table()
        return analysis

    def _load_ulysses(self) -> tuple[dict, list]:
        from mestrado.data.loader import DataLoader
        loader = DataLoader()
        bills, queries = loader.load_all()
        logger.info(f"Ulysses: {len(bills):,} bills, {len(queries):,} queries loaded.")
        return bills, queries

    def _load_miracl(self) -> tuple[dict, list]:
        from mestrado.data.miracl_loader import load_miracl_pt, describe_miracl_pt
        bills, queries = load_miracl_pt(
            split=cfg.miracl_split,
            max_queries=self.n_queries * 5 if self.n_queries else None,
        )
        stats = describe_miracl_pt(bills, queries)
        logger.info(f"MIRACL PT: {stats}")
        return bills, queries

    @staticmethod
    def save(analysis: CrossDomainAnalysis, results_dir: Path) -> Path:
        """Save full cross-domain analysis JSON."""
        results_dir.mkdir(exist_ok=True)
        ts = analysis.timestamp.replace(":", "-").replace("T", "_")
        out_path = results_dir / f"cross_domain_{ts}.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(analysis.to_dict(), f, ensure_ascii=False, indent=2)

        # Save per-dataset results too
        if analysis.ulysses_results:
            SLMComparisonRunner.save(analysis.ulysses_results, results_dir)
        if analysis.miracl_results:
            SLMComparisonRunner.save(analysis.miracl_results, results_dir)

        logger.info(f"Cross-domain analysis saved to {out_path}")
        return out_path

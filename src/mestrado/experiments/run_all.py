"""
CLI entry point for Ulysses-MTRAG experiments.

Single-turn experiments (E1–E8):
    uv sync
    uv run run-experiments --help
    uv run run-experiments                              # E1–E3
    uv run run-experiments --experiments e1 e3 e4
    uv run run-experiments --experiments e1 e2 e3 e4 e5 --generation
    uv run run-experiments --sample 10                  # quick test
    uv run run-experiments --full                       # all 692 queries

Multi-turn dataset generation (Ulysses-MTRAG):
    uv run run-experiments generate-mtrag               # 100 sessions, 4 turns
    uv run run-experiments generate-mtrag --n-sessions 200 --n-turns 5
    uv run run-experiments generate-mtrag --output data/ulysses_mtrag.json

Multi-turn evaluation (after generating):
    uv run run-experiments eval-multiturn --dataset data/ulysses_mtrag.json
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="run-experiments",
    help="BillsRAG-BR benchmark experiments runner",
    add_completion=False,
)
# Configure console for Windows compatibility
console = Console(force_terminal=True, legacy_windows=False)


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def main(
    experiments: Optional[list[str]] = typer.Option(
        None,
        "--experiments", "-e",
        help="Experiment IDs to run: e1 e2 e3 e4 e5 e6 e7 e8 (default: e1 e2 e3)",
    ),
    sample: Optional[int] = typer.Option(
        None,
        "--sample", "-n",
        help="Number of queries to use (overrides EXPERIMENT_SAMPLE_SIZE in .env)",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run on all 692 queries (ignores --sample)",
    ),
    generation: bool = typer.Option(
        False,
        "--generation",
        help="Run generation experiments E6/E7/E8 (requires Ollama)",
    ),
    results_dir: Optional[Path] = typer.Option(
        None,
        "--results-dir",
        help="Directory to save results (default: results/)",
    ),
    log_level: str = typer.Option(
        "INFO",
        "--log-level",
        help="Logging level: DEBUG, INFO, WARNING",
    ),
) -> None:
    """Run BillsRAG-BR retrieval and generation experiments."""
    _setup_logging(log_level)
    logger = logging.getLogger(__name__)

    from mestrado.experiments.runner import ExperimentRunner
    from mestrado.config import experiment as exp_cfg, output as out_cfg

    # Resolve parameters
    effective_experiments = experiments or ["e1", "e2", "e3"]
    effective_sample = None if full else (sample or exp_cfg.sample_size)
    effective_results_dir = results_dir or out_cfg.results_dir

    console.rule("[bold blue]BillsRAG-BR Experiment Runner")
    console.print(f"Experiments: {effective_experiments}")
    console.print(f"Queries: {'ALL' if full else effective_sample}")
    console.print(f"Generation: {generation}")
    console.print(f"Results dir: {effective_results_dir}")
    console.rule()

    runner = ExperimentRunner(
        sample_size=effective_sample,
        run_generation=generation,
        results_dir=effective_results_dir,
    )

    # Setup: always build BM25; dense only if needed
    runner.setup()

    if any(e in effective_experiments for e in ["e2", "e3", "e4"]):
        runner.setup_dense()

    console.rule("[bold green]Running experiments")
    results = runner.run_all(experiments=effective_experiments)

    console.rule("[bold yellow]Results")
    out_path = runner.save_and_report(results)

    console.print(f"\n[bold green]Done! Results saved to {out_path}")
    console.print("\nNext steps:")
    console.print("  • Open results/all_results.json for detailed per-query metrics")
    console.print("  • Check results/significance_tests.json for H1–H5 verdicts")
    if generation:
        console.print("  • Check results/e6_generation_results.json for RAG outputs")


@app.command("check-datasets")
def check_datasets() -> None:
    """Validate dataset loading and join integrity."""
    _setup_logging()
    from mestrado.data.loader import DataLoader
    loader = DataLoader()
    bills, queries = loader.load_all()
    console.print(f"[green]Bills loaded: {len(bills):,}")
    console.print(f"[green]Queries loaded: {len(queries)}")
    console.print(f"[green]Valid bill names: {len(loader.get_valid_bill_names()):,}")


@app.command("check-ollama")
def check_ollama() -> None:
    """Verify Ollama is running and list available models."""
    _setup_logging()
    try:
        import ollama
        from mestrado.config import ollama as cfg
        models = ollama.list()
        model_names = [m.get("name", m) for m in models.get("models", [])]
        console.print(f"[green]Ollama connected at {cfg.base_url}")
        console.print(f"Available models: {model_names}")

        if cfg.reranker_model not in model_names:
            console.print(
                f"[yellow]Warning: reranker model '{cfg.reranker_model}' not found. "
                f"Run: ollama pull {cfg.reranker_model}"
            )
        if cfg.generator_model not in model_names:
            console.print(
                f"[yellow]Warning: generator model '{cfg.generator_model}' not found. "
                f"Run: ollama pull {cfg.generator_model}"
            )
    except Exception as e:
        console.print(f"[red]Ollama not available: {e}")
        console.print("Start Ollama: ollama serve")
        sys.exit(1)


@app.command("quick-test")
def quick_test() -> None:
    """Run a minimal smoke test (5 queries, E1 only) to verify setup."""
    _setup_logging("DEBUG")
    console.print("[bold]Running smoke test with 5 queries, E1 (BM25) only...")

    from mestrado.experiments.runner import ExperimentRunner
    runner = ExperimentRunner(sample_size=5)
    runner.setup()
    results = runner.run_all(experiments=["e1"])
    runner.save_and_report(results)
    console.print("[bold green]Smoke test passed!")


@app.command("generate-mtrag")
def generate_mtrag(
    n_sessions: int = typer.Option(100, "--n-sessions", help="Number of conversations to generate"),
    n_turns: int = typer.Option(4, "--n-turns", help="Number of turns per conversation"),
    generator_model: str = typer.Option("llama3.1:8b", "--generator", help="Ollama model for generation"),
    judge_model: str = typer.Option("phi3:medium", "--judge", help="Ollama model for relevance judging"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output JSON path (default: data/ulysses_mtrag.json)"
    ),
    seed_min_relevant: int = typer.Option(3, "--min-relevant", help="Min relevant bills per seed query"),
) -> None:
    """
    Generate Ulysses-MTRAG synthetic multi-turn dataset.

    Extends Ulysses-RFCorpus (Vitório et al., LRE 2024) with multi-turn conversations.
    Methodology: MTRAG-synthetic (Katsis et al., TACL 2025).
    Turn 1: ground truth from 54 Conle specialists (human).
    Turns 2+: LLM-generated queries + LLM-as-judge relevance.
    """
    _setup_logging()
    console.rule("[bold blue]Ulysses-MTRAG Generator")
    console.print(f"Sessions: {n_sessions} | Turns: {n_turns}")
    console.print(f"Generator: {generator_model} | Judge: {judge_model}")

    from mestrado.data.loader import DataLoader
    from mestrado.synthetic.generator import UlyssesMTRAGGenerator

    loader = DataLoader()
    _, queries = loader.load_all()

    output_path = output or Path("data/ulysses_mtrag.json")

    generator = UlyssesMTRAGGenerator(
        generator_model=generator_model,
        judge_model=judge_model,
        seed_n_relevant_min=seed_min_relevant,
        n_turns=n_turns,
    )

    embeddings_cache = Path("data/.mtrag_seed_embeddings.pkl")
    conversations = generator.generate_dataset(
        queries=queries,
        n_sessions=n_sessions,
        output_path=output_path,
        embeddings_cache=embeddings_cache,
    )

    console.print(f"[bold green]Generated {len(conversations)} conversations → {output_path}")
    console.print("\nNext steps:")
    console.print(f"  • Review generated dataset: {output_path}")
    console.print("  • Run multi-turn evaluation: uv run run-experiments eval-multiturn")


@app.command("eval-multiturn")
def eval_multiturn(
    dataset: Path = typer.Argument(..., help="Path to ulysses_mtrag.json"),
    system: str = typer.Option("bm25", "--system", help="Retrieval system: bm25|dense|hybrid|reranker"),
    judge_model: str = typer.Option("phi3:medium", "--judge", help="LLM for relevance judging"),
    k: int = typer.Option(12, "--k", help="Number of retrieved documents"),
    output: Optional[Path] = typer.Option(None, "--output", "-o"),
) -> None:
    """
    Evaluate retrieval system on Ulysses-MTRAG multi-turn dataset.

    Measures nDCG progression over turns — key evidence for multi-turn benefit.
    """
    _setup_logging()
    import json
    from mestrado.metrics.multiturn_metrics import evaluate_turn, evaluate_session, aggregate_session_results
    from mestrado.synthetic.session_context import SessionContextManager

    console.rule("[bold blue]Ulysses-MTRAG Multi-Turn Evaluation")
    console.print(f"Dataset: {dataset} | System: {system} | K: {k}")

    if not dataset.exists():
        console.print(f"[red]Dataset not found: {dataset}")
        console.print("Run: uv run run-experiments generate-mtrag")
        raise typer.Exit(1)

    with open(dataset, encoding="utf-8") as f:
        data = json.load(f)

    conversations_data = data.get("conversations", [])
    console.print(f"Loaded {len(conversations_data)} conversations")

    # Setup retrieval system
    from mestrado.data.loader import DataLoader
    from mestrado.retrieval.bm25 import BM25Retriever
    from mestrado.retrieval.dense import DenseRetriever
    from mestrado.retrieval.hybrid import HybridRetriever
    loader = DataLoader()
    bills, _ = loader.load_all()

    if system == "dense":
        retriever = DenseRetriever(bills)
        retriever.build_index()
    elif system in ("hybrid", "reranker"):
        bm25 = BM25Retriever(bills)
        bm25.build_index()
        dense = DenseRetriever(bills)
        dense.build_index()
        retriever = HybridRetriever(bm25, dense)
    else:  # default: bm25
        retriever = BM25Retriever(bills)
        retriever.build_index()

    ctx_manager = SessionContextManager()
    all_session_results = []

    for conv_data in conversations_data:
        session_id = conv_data["session_id"]
        ctx_manager.reset()
        turn_results = []

        for turn_data in conv_data["turns"]:
            query = turn_data["query"]
            turn_n = turn_data["n"]
            turn_type = turn_data["type"]

            # Resolve nonstandalone queries using session context
            if turn_n > 1:
                resolved_query = ctx_manager.resolve_query(query)
            else:
                resolved_query = query

            # Retrieve
            retrieved = retriever.retrieve(resolved_query, k=k)
            retrieved_names = [b.name for b in retrieved]

            # Use original judgments from dataset if available, else empty
            judgments = turn_data.get("judgments", {})
            if not judgments and turn_n == 1:
                # Use Ulysses-RFCorpus ground truth for turn 1
                # (would need to look up seed query — simplified here)
                judgments = {name: 0.0 for name in retrieved_names}

            result = evaluate_turn(retrieved_names, judgments, turn_n, turn_type, query, k)
            turn_results.append(result)

            # Update context manager for next turn
            from mestrado.synthetic.generator import Turn as GenTurn
            ctx_manager.add_turn(GenTurn(n=turn_n, type=turn_type, query=query))

        all_session_results.append(evaluate_session(session_id, turn_results))

    # Aggregate and report
    aggregated = aggregate_session_results(all_session_results)

    console.print("\n[bold]nDCG@12 by Turn:")
    for turn_pos, ndcg in aggregated.get("mean_ndcg_by_turn", {}).items():
        console.print(f"  {turn_pos}: {ndcg:.4f}")

    console.print("\n[bold]Mean Gain per Turn (Δ nDCG):")
    for transition, gain in aggregated.get("mean_gain_per_turn", {}).items():
        color = "green" if gain > 0 else "red"
        console.print(f"  [{color}]{transition}: {gain:+.4f}[/{color}]")

    rate = aggregated.get("improvement_rate", 0)
    console.print(f"\n[bold]Sessions improved by multi-turn: {rate:.1%}")

    if output:
        import json
        with open(output, "w") as f:
            json.dump(aggregated, f, indent=2)
        console.print(f"\nResults saved to {output}")


@app.command("slm-compare")
def slm_compare(
    n_queries: Optional[int] = typer.Option(
        None,
        "--n-queries", "-n",
        help="Number of queries (default: SLM_N_QUERIES in .env, usually 100)",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run on all available queries (692 for Ulysses)",
    ),
    oracle_only: bool = typer.Option(
        False,
        "--oracle-only",
        help="Run only the oracle LLM (validate prompts, estimate cost before SLMs)",
    ),
    dataset: str = typer.Option(
        "ulysses",
        "--dataset",
        help="Dataset to run on: ulysses (default)",
    ),
    results_dir: Optional[Path] = typer.Option(
        None,
        "--results-dir",
        help="Directory to save results (default: results/)",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """
    Run SLM comparison experiment — 4 SLMs + oracle LLM on Ulysses legislative corpus.

    Compares Phi-4 Mini, Gemma 3 4B, Qwen2.5 7B, Llama 3.1 8B (SLMs) against
    Llama 3.3 70B (oracle) for RAG generation quality. Evaluates ROUGE-L,
    BERTScore, and faithfulness (LLM-as-judge). Addresses hypothesis H5.

    Examples:

        uv run run-experiments slm-compare --n-queries 5 --oracle-only  # smoke test

        uv run run-experiments slm-compare --n-queries 100              # validation

        uv run run-experiments slm-compare --full                       # all 692 queries
    """
    _setup_logging(log_level)
    from mestrado.experiments.slm_comparison import SLMComparisonRunner
    from mestrado.data.loader import DataLoader
    from mestrado.config import slm_comparison as slm_cfg, output as out_cfg

    effective_n = None if full else n_queries
    effective_results_dir = results_dir or out_cfg.results_dir

    console.rule("[bold blue]SLM Comparison Experiment")
    console.print(f"Dataset: {dataset.upper()}")
    console.print(f"Queries: {'ALL' if full else (effective_n or slm_cfg.n_queries)}")
    console.print(f"Oracle only: {oracle_only}")
    console.print(f"Oracle model: {slm_cfg.oracle_model}")
    if not oracle_only:
        console.print(f"SLM models: {slm_cfg.slm_models}")
    console.print(f"Results dir: {effective_results_dir}")
    console.rule()

    loader = DataLoader()
    bills, queries = loader.load_all()

    runner = SLMComparisonRunner(
        bills_by_name=bills,
        queries=queries,
        dataset_name=dataset,
        n_queries=effective_n,
        seed=slm_cfg.seed,
    ).setup()

    results = runner.run(oracle_only=oracle_only)
    out_path = SLMComparisonRunner.save(results, effective_results_dir)
    console.print(f"\n[bold green]Done! Results saved to {out_path}")


@app.command("cross-domain")
def cross_domain(
    n_queries: Optional[int] = typer.Option(
        None,
        "--n-queries", "-n",
        help="Number of queries per dataset (default: SLM_N_QUERIES in .env)",
    ),
    full: bool = typer.Option(
        False,
        "--full",
        help="Run on all queries (692 Ulysses + ~1k MIRACL)",
    ),
    oracle_only: bool = typer.Option(
        False,
        "--oracle-only",
        help="Run only the oracle LLM on both datasets (validate + estimate cost)",
    ),
    results_dir: Optional[Path] = typer.Option(
        None,
        "--results-dir",
        help="Directory to save results (default: results/)",
    ),
    log_level: str = typer.Option("INFO", "--log-level"),
) -> None:
    """
    Cross-domain SLM comparison — Ulysses (legislative) vs. MIRACL (Wikipedia PT-BR).

    Runs the same SLM comparison pipeline on both datasets to quantify the domain gap.
    Research question: do SLMs suffer more than LLMs from domain specialization?

    References:
      Ulysses: Vitório et al. (LRE 2024) DOI:10.1007/s10579-024-09767-3
      MIRACL:  Zhang et al. (2022) arXiv:2209.05299

    Examples:

        uv run run-experiments cross-domain --n-queries 5 --oracle-only   # smoke test

        uv run run-experiments cross-domain --n-queries 100               # validation

        uv run run-experiments cross-domain --full                        # complete run
    """
    _setup_logging(log_level)
    from mestrado.experiments.cross_domain import CrossDomainRunner
    from mestrado.config import slm_comparison as slm_cfg, output as out_cfg

    effective_n = None if full else n_queries
    effective_results_dir = results_dir or out_cfg.results_dir

    console.rule("[bold blue]Cross-Domain SLM Experiment")
    console.print("Datasets: Ulysses-RFCorpus (legislative) + MIRACL Portuguese (Wikipedia)")
    console.print(f"Queries per dataset: {'ALL' if full else (effective_n or slm_cfg.n_queries)}")
    console.print(f"Oracle only: {oracle_only}")
    console.rule()

    runner = CrossDomainRunner(n_queries=effective_n, seed=slm_cfg.seed)
    analysis = runner.run(oracle_only=oracle_only)
    out_path = CrossDomainRunner.save(analysis, effective_results_dir)
    console.print(f"\n[bold green]Done! Cross-domain analysis saved to {out_path}")
    console.print("\nKey metric: Domain Gap = (MIRACL - Ulysses) / MIRACL")
    console.print("  Positive gap → model degrades in specialized legislative domain")
    console.print("  Near-zero gap → model generalizes well across domains (cf. Belcak & Wattenhofer 2025)")


if __name__ == "__main__":
    app()

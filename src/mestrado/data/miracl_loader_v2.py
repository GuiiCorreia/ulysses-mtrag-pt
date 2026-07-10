"""
MIRACL Portuguese loader v2 — cross-domain evaluation dataset.

Updated version that loads MIRACL datasets from HuggingFace Hub
using the new Parquet format (no scripts required).

MIRACL Portuguese:
  - ~60k Wikipedia PT-BR passages as corpus
  - ~1k queries with binary relevance judgments (dev split)
  - Queries are information-seeking, natural language PT-BR

Reference:
  Zhang, X., Thakur, N., Ogundepo, O., Maia, E., Wakatsuki, E., Boualili, L.,
  Jafari, F., Ma, X., Pradeep, R., & Lin, J. (2022). MIRACL: A Multilingual
  Retrieval Dataset Covering 18 Diverse Languages. arXiv:2209.05299.

Usage:
    from mestrado.data.miracl_loader_v2 import load_miracl_pt_v2
    bills, queries = load_miracl_pt_v2(split="dev", max_corpus=60000)
"""
from __future__ import annotations

import logging
from typing import Optional

from mestrado.data.schema import Bill, FeedbackItem, Query

logger = logging.getLogger(__name__)

# Sentinel sig_tipo to distinguish MIRACL passages from Ulysses bills
MIRACL_SIG_TIPO = "MIRACL-PT"


def load_miracl_pt_v2(
    split: str = "dev",
    max_corpus: Optional[int] = None,
    max_queries: Optional[int] = None,
) -> tuple[dict[str, Bill], list[Query]]:
    """
    Load MIRACL Portuguese dataset using direct Parquet loading.

    Args:
        split: Dataset split — "dev" has qrels (relevance judgments available).
        max_corpus: Limit corpus size for smoke tests. None = full ~60k passages.
        max_queries: Limit queries for smoke tests. None = full ~1k dev queries.

    Returns:
        bills_by_name: dict[passage_id → Bill] (passages as "bills")
        queries: list[Query] with relevance judgments from MIRACL qrels
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Install pandas: uv add pandas")

    # Load corpus from HuggingFace Hub (Parquet format)
    logger.info("Loading MIRACL Portuguese corpus from HuggingFace Hub...")
    corpus_url = "https://huggingface.co/datasets/miracl/miracl-corpus/resolve/main/pt/train-00000-of-00001.parquet"

    try:
        corpus_df = pd.read_parquet(corpus_url)
    except Exception as e:
        logger.warning(f"Could not load from HuggingFace Hub: {e}")
        logger.info("Falling back to local dataset cache...")

        # Try loading from datasets library (might work with newer versions)
        from datasets import load_dataset
        corpus_ds = load_dataset("miracl/miracl-corpus", "pt", split="train")
        corpus_df = corpus_ds.to_pandas()

    if max_corpus:
        corpus_df = corpus_df.head(max_corpus)

    logger.info(f"MIRACL corpus: {len(corpus_df):,} passages loaded.")

    # Build bills_by_name: passage_id → Bill
    bills_by_name: dict[str, Bill] = {}
    for _, row in corpus_df.iterrows():
        docid = str(row.get("docid", ""))
        title = str(row.get("title", "") or "")
        text = str(row.get("text", "") or "")

        if not docid:
            continue

        # Use title as "name" (join key), text as both ementa and full text
        ementa = (title + " — " + text[:400]).strip() if title else text[:500]

        bill = Bill(
            code=docid,
            sig_tipo=MIRACL_SIG_TIPO,
            name=docid,
            txt_ementa=ementa,
            em_tramitacao="Sim",
            situacao="",
            text=text,
            preprocessed_tokens=text.lower().split(),
        )
        bills_by_name[docid] = bill

    logger.info(f"Bills map built: {len(bills_by_name):,} passages.")

    # Load queries with qrels
    logger.info(f"Loading MIRACL Portuguese '{split}' queries + qrels...")
    # MIRACL queries are typically available in the main dataset
    queries_url = f"https://huggingface.co/datasets/miracl/miracl/resolve/main/pt/{split}-00000-of-00001.parquet"

    try:
        queries_df = pd.read_parquet(queries_url)
    except Exception as e:
        logger.warning(f"Could not load queries from HuggingFace Hub: {e}")
        logger.info("Trying datasets library...")

        from datasets import load_dataset
        query_ds = load_dataset("miracl/miracl", "pt", split=split)
        queries_df = query_ds.to_pandas()

    if max_queries:
        queries_df = queries_df.head(max_queries)

    queries: list[Query] = []
    for _, row in queries_df.iterrows():
        qid = str(row.get("query_id", ""))
        qtext = str(row.get("query", "") or "")

        # Get positive and negative passages
        pos_passages = row.get("positive_passages", [])
        neg_passages = row.get("negative_passages", [])

        # Handle different formats
        if isinstance(pos_passages, list):
            pos_passages = [p if isinstance(p, dict) else {"docid": p} for p in pos_passages]
        if isinstance(neg_passages, list):
            neg_passages = [n if isinstance(n, dict) else {"docid": n} for n in neg_passages]

        # Build FeedbackItems
        feedback: list[FeedbackItem] = []
        for p in pos_passages:
            did = str(p.get("docid", "")) if isinstance(p, dict) else str(p)
            if did in bills_by_name:
                feedback.append(FeedbackItem(
                    bill_id=did,
                    label="r",
                    score=1.0,
                    score_normalized=1.0,
                ))

        for n in neg_passages:
            did = str(n.get("docid", "")) if isinstance(n, dict) else str(n)
            if did in bills_by_name:
                feedback.append(FeedbackItem(
                    bill_id=did,
                    label="i",
                    score=0.0,
                    score_normalized=0.0,
                ))

        if not feedback:
            continue

        queries.append(Query(
            query_id=qid,
            text=qtext,
            feedback=feedback,
            extra_results=[],
            date_created="",
            num_doc_feedback=len(feedback),
        ))

    logger.info(f"Queries loaded: {len(queries):,} with at least 1 judgment.")
    return bills_by_name, queries


def describe_miracl_pt_v2(bills: dict[str, Bill], queries: list[Query]) -> dict:
    """Return basic statistics about the loaded MIRACL PT dataset."""
    n_relevant = sum(
        sum(1 for f in q.feedback if f.label == "r")
        for q in queries
    )
    n_total_judgments = sum(len(q.feedback) for q in queries)
    return {
        "n_passages": len(bills),
        "n_queries": len(queries),
        "n_total_judgments": n_total_judgments,
        "n_relevant": n_relevant,
        "avg_relevant_per_query": round(n_relevant / max(1, len(queries)), 2),
        "dataset": "MIRACL Portuguese (Zhang et al. 2022, arXiv:2209.05299)",
        "domain": "Wikipedia PT-BR (general domain)",
    }

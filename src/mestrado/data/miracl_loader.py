"""
MIRACL Portuguese loader — cross-domain evaluation dataset.

Loads MIRACL Portuguese (Wikipedia PT-BR) and maps it to the existing
Bill/Query schema, enabling cross-domain comparison against Ulysses-RFCorpus.

MIRACL Portuguese:
  - ~60k Wikipedia PT-BR passages as corpus
  - ~1k queries with binary relevance judgments (dev split)
  - Queries are information-seeking, natural language PT-BR

Used in cross-domain experiments to test whether:
  "SLMs 2025 perform differently on specialized (legislative) vs. general (Wikipedia)
   PT-BR retrieval?" — answering the domain gap hypothesis.

Reference:
  Zhang, X., Thakur, N., Ogundepo, O., Maia, E., Wakatsuki, E., Boualili, L.,
  Jafari, F., Ma, X., Pradeep, R., & Lin, J. (2022). MIRACL: A Multilingual
  Retrieval Dataset Covering 18 Diverse Languages. arXiv:2209.05299.

Usage:
    from mestrado.data.miracl_loader import load_miracl_pt
    bills, queries = load_miracl_pt(split="dev", max_corpus=60000)
"""
from __future__ import annotations

import logging
from typing import Optional

from mestrado.data.schema import Bill, FeedbackItem, Query

logger = logging.getLogger(__name__)

# Sentinel sig_tipo to distinguish MIRACL passages from Ulysses bills
MIRACL_SIG_TIPO = "MIRACL-PT"


def load_miracl_pt(
    split: str = "dev",
    max_corpus: Optional[int] = None,
    max_queries: Optional[int] = None,
) -> tuple[dict[str, Bill], list[Query]]:
    """
    Load MIRACL Portuguese dataset and adapt to Bill/Query schema.

    Args:
        split: Dataset split — "dev" has qrels (relevance judgments available).
               "train" is larger but lacks public qrels.
        max_corpus: Limit corpus size for smoke tests. None = full ~60k passages.
        max_queries: Limit queries for smoke tests. None = full ~1k dev queries.

    Returns:
        bills_by_name: dict[passage_id → Bill] (passages as "bills")
        queries: list[Query] with relevance judgments from MIRACL qrels

    Raises:
        ImportError: If `datasets` package not installed.
        ValueError: If the requested split has no qrels.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install datasets package: uv add datasets"
        )

    if split == "train":
        raise ValueError(
            "MIRACL 'train' split has no public qrels. Use split='dev'."
        )

    logger.info("Loading MIRACL Portuguese corpus...")
    corpus_ds = load_dataset(
        "miracl/miracl-corpus",
        "pt",
        trust_remote_code=True,
    )["train"]

    if max_corpus:
        corpus_ds = corpus_ds.select(range(min(max_corpus, len(corpus_ds))))

    logger.info(f"MIRACL corpus: {len(corpus_ds):,} passages loaded.")

    # Build bills_by_name: passage_id → Bill
    # Mapping: docid → code, title → name, text → txt_ementa + text
    bills_by_name: dict[str, Bill] = {}
    for row in corpus_ds:
        docid = str(row["docid"])
        title = str(row.get("title", "") or "")
        text = str(row.get("text", "") or "")

        # Use title as "name" (join key), text as both ementa and full text
        # For short passages (Wikipedia), ementa = first 500 chars of text
        ementa = (title + " — " + text[:400]).strip() if title else text[:500]

        bill = Bill(
            code=docid,
            sig_tipo=MIRACL_SIG_TIPO,
            name=docid,           # docid is the join key (no bill code format)
            txt_ementa=ementa,
            em_tramitacao="Sim",  # not applicable, use default
            situacao="",
            text=text,
            preprocessed_tokens=text.lower().split(),
        )
        bills_by_name[docid] = bill

    logger.info(f"Bills map built: {len(bills_by_name):,} passages.")

    # Load queries with qrels
    logger.info(f"Loading MIRACL Portuguese '{split}' queries + qrels...")
    query_ds = load_dataset(
        "miracl/miracl",
        "pt",
        trust_remote_code=True,
    )[split]

    if max_queries:
        query_ds = query_ds.select(range(min(max_queries, len(query_ds))))

    queries: list[Query] = []
    for row in query_ds:
        qid = str(row["query_id"])
        qtext = str(row.get("query", "") or "")

        # positive_passages: list of {"docid": ..., "text": ...}
        # negative_passages: list of {"docid": ..., "text": ...} (hard negatives)
        pos_passages = row.get("positive_passages", []) or []
        neg_passages = row.get("negative_passages", []) or []

        # Build FeedbackItems: positives = "r" (1.0), negatives = "i" (0.0)
        feedback: list[FeedbackItem] = []
        for p in pos_passages:
            did = str(p.get("docid", ""))
            if did in bills_by_name:
                feedback.append(FeedbackItem(
                    bill_id=did,
                    label="r",
                    score=1.0,
                    score_normalized=1.0,
                ))
        for n in neg_passages:
            did = str(n.get("docid", ""))
            if did in bills_by_name:
                feedback.append(FeedbackItem(
                    bill_id=did,
                    label="i",
                    score=0.0,
                    score_normalized=0.0,
                ))

        if not feedback:
            # Skip queries with no relevant docs in corpus (corpus subset case)
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


def describe_miracl_pt(bills: dict[str, Bill], queries: list[Query]) -> dict:
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

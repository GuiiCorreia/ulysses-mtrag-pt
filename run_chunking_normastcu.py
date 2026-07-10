"""
Chunking experiment for NormasTCU (ICTAI 2026).

Goal: test whether passage-level (chunk) indexing recovers dense retrieval on
NormasTCU, where whole-document dense retrieval collapses (nDCG@10 ~ 0).

This script is STANDALONE and does NOT modify any existing module. It reuses
BEIRLoader, BM25Retriever, DenseRetriever, and ir_metrics, building variant
Bill dicts in-memory to control exactly which text each retriever sees.

It runs and compares these retrieval conditions on the SAME qrels:

  bm25            BM25 over the full document text            (reference)
  dense_summary   BGE-M3 over txt_ementa = ASSUNTO summary    (BEIRLoader default baseline)
  dense_fulldoc   BGE-M3 over the full doc text (BGE truncates at 8192 tokens)
                                                              (the "truncation" baseline)
  dense_chunked   BGE-M3 over <chunk-size passages of the full doc,
                  mapped back to parent docs                  (the proposed fix)

Why three dense conditions: it disentangles the cause of the dense failure
(summary-only vs. 8192-token truncation) and shows whether chunking fixes it.

Usage (run from codigos_mestrado/):
    uv run python run_chunking_normastcu.py
    uv run python run_chunking_normastcu.py --chunk-size 3000 --overlap 400 --max-doc-chars 30000
    uv run python run_chunking_normastcu.py --conditions bm25 dense_summary dense_fulldoc
    uv run python run_chunking_normastcu.py --conditions dense_chunked   # heaviest; cached after 1st run

Output: prints a comparison table and writes results/chunking_normastcu.json
"""
import sys
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mestrado.data.beir_loader import BEIRLoader
from mestrado.data.schema import Bill
from mestrado.retrieval.bm25 import BM25Retriever
from mestrado.retrieval.dense import DenseRetriever
from mestrado.metrics.ir_metrics import ndcg_at_k, mrr, recall_at_k, average_precision

BGE_M3_MODEL = "BAAI/bge-m3"
CACHE_DIR    = Path("results/cache/chunking")
OUT_PATH     = Path("results/chunking_normastcu.json")

ALL_CONDITIONS = ["bm25", "dense_summary", "dense_fulldoc", "dense_chunked"]


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding-window character chunks with overlap. Robust and model-agnostic."""
    if not text:
        return []
    step = max(1, size - overlap)
    chunks = [text[i:i + size] for i in range(0, len(text), step)]
    return [c for c in chunks if c.strip()]


def build_chunk_bills(bills: dict[str, Bill], size: int, overlap: int):
    """Return (chunk_bills_by_name, chunk_to_parent). Chunk text goes in BOTH
    txt_ementa (dense embeds this) and text (BM25 indexes this)."""
    chunk_bills: dict[str, Bill] = {}
    chunk_to_parent: dict[str, str] = {}
    for name, bill in bills.items():
        chunks = chunk_text(bill.text or bill.txt_ementa, size, overlap)
        for j, ch in enumerate(chunks):
            cname = f"{name}#c{j}"
            chunk_bills[cname] = Bill(
                code=name, sig_tipo=bill.sig_tipo, name=cname,
                txt_ementa=ch, em_tramitacao="", situacao="", text=ch,
            )
            chunk_to_parent[cname] = name
    return chunk_bills, chunk_to_parent


def clone_with_ementa_as_fulltext(bills: dict[str, Bill]) -> dict[str, Bill]:
    """Variant where dense will embed the full doc text (BGE truncates at 8192)."""
    out: dict[str, Bill] = {}
    for name, bill in bills.items():
        out[name] = Bill(
            code=bill.code, sig_tipo=bill.sig_tipo, name=bill.name,
            txt_ementa=(bill.text or bill.txt_ementa),  # dense embeds this
            em_tramitacao=bill.em_tramitacao, situacao=bill.situacao, text=bill.text,
        )
    return out


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def doc_ranking_from_chunks(chunk_results, chunk_to_parent: dict[str, str],
                            top_docs: int) -> list[str]:
    """Collapse a ranked list of chunks into a ranked list of parent docs
    (first/best-ranked chunk per doc wins)."""
    seen, doc_ids = set(), []
    for r in chunk_results:
        parent = chunk_to_parent.get(r.bill.name, r.bill.name)
        if parent not in seen:
            seen.add(parent)
            doc_ids.append(parent)
        if len(doc_ids) >= top_docs:
            break
    return doc_ids


def score(queries, ranking_fn) -> dict:
    """ranking_fn(query) -> ordered list of doc ids. Computes IR metrics."""
    n5, n10, mrr_l, r10, ap = [], [], [], [], []
    for q in queries:
        g = q.graded_relevance()
        ids = ranking_fn(q)
        n5.append(ndcg_at_k(ids, g, k=5))
        n10.append(ndcg_at_k(ids, g, k=10))
        mrr_l.append(mrr(ids, g))
        r10.append(recall_at_k(ids, g, k=10))
        ap.append(average_precision(ids, g))
    avg = lambda l: round(sum(l) / len(l), 4) if l else 0.0
    return {"nDCG@5": avg(n5), "nDCG@10": avg(n10), "MRR": avg(mrr_l),
            "Recall@10": avg(r10), "MAP": avg(ap)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="NormasTCU chunking experiment (ICTAI 2026)")
    ap.add_argument("--max-doc-chars", type=int, default=30_000,
                    help="Truncate each norm to N chars before chunking (default 30k)")
    ap.add_argument("--chunk-size", type=int, default=3000,
                    help="Chunk size in characters (~750 tokens; well under BGE 8192)")
    ap.add_argument("--overlap", type=int, default=400, help="Chunk overlap in chars")
    ap.add_argument("--top-docs", type=int, default=10, help="Docs to keep for metrics")
    ap.add_argument("--chunk-pool", type=int, default=100,
                    help="Top chunks retrieved before collapsing to docs")
    ap.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                    choices=ALL_CONDITIONS)
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*70}\nNORMASTCU CHUNKING EXPERIMENT\n{'='*70}")
    print(f"max_doc_chars={args.max_doc_chars}  chunk_size={args.chunk_size}  "
          f"overlap={args.overlap}  conditions={args.conditions}")

    # Load corpus (full text up to max_doc_chars so chunking sees real content)
    loader = BEIRLoader("normas-tcu", max_doc_chars=args.max_doc_chars)
    bills, queries = loader.load()
    print(f"  corpus={len(bills):,} docs  queries={len(queries)}")

    results: dict = {}

    # --- BM25 (full text) -------------------------------------------------
    if "bm25" in args.conditions:
        print("\n[bm25] indexing full text ...")
        ret = BM25Retriever(bills); ret.build_index()
        fn = lambda q: [r.bill.name for r in ret.retrieve(q.text, top_k=args.top_docs)]
        results["bm25"] = score(queries, fn)
        print(f"  {results['bm25']}")

    # --- Dense: summary only (BEIRLoader default) -------------------------
    if "dense_summary" in args.conditions:
        print("\n[dense_summary] embedding txt_ementa (ASSUNTO summary) ...")
        ret = DenseRetriever(bills, model_name=BGE_M3_MODEL)
        ret.build_index(cache_path=CACHE_DIR / "dense_summary.pkl")
        fn = lambda q: [r.bill.name for r in ret.retrieve(q.text, top_k=args.top_docs)]
        results["dense_summary"] = score(queries, fn)
        print(f"  {results['dense_summary']}")

    # --- Dense: full doc (BGE truncates at 8192 tokens) -------------------
    if "dense_fulldoc" in args.conditions:
        print("\n[dense_fulldoc] embedding full doc text (8192-token BGE cap applies) ...")
        full_bills = clone_with_ementa_as_fulltext(bills)
        ret = DenseRetriever(full_bills, model_name=BGE_M3_MODEL)
        ret.build_index(cache_path=CACHE_DIR / "dense_fulldoc.pkl")
        fn = lambda q: [r.bill.name for r in ret.retrieve(q.text, top_k=args.top_docs)]
        results["dense_fulldoc"] = score(queries, fn)
        print(f"  {results['dense_fulldoc']}")

    # --- Dense: chunked (the proposed fix) --------------------------------
    if "dense_chunked" in args.conditions:
        print("\n[dense_chunked] chunking + embedding passages ...")
        chunk_bills, chunk_to_parent = build_chunk_bills(bills, args.chunk_size, args.overlap)
        print(f"  built {len(chunk_bills):,} chunks from {len(bills):,} docs "
              f"(~{len(chunk_bills)//max(1,len(bills))}/doc)")
        ret = DenseRetriever(chunk_bills, model_name=BGE_M3_MODEL)
        cache = CACHE_DIR / f"dense_chunked_{args.max_doc_chars}_{args.chunk_size}_{args.overlap}.pkl"
        ret.build_index(cache_path=cache)
        def fn(q):
            chunks = ret.retrieve(q.text, top_k=args.chunk_pool)
            return doc_ranking_from_chunks(chunks, chunk_to_parent, args.top_docs)
        results["dense_chunked"] = score(queries, fn)
        print(f"  {results['dense_chunked']}")

    # --- Summary table ----------------------------------------------------
    print(f"\n{'='*70}\nRESULTS  (NormasTCU, n={len(queries)} queries)\n{'='*70}")
    print(f"{'condition':<16}{'nDCG@5':>9}{'nDCG@10':>9}{'MRR':>8}{'Recall@10':>11}{'MAP':>8}")
    print("-" * 70)
    for cond in args.conditions:
        if cond in results:
            r = results[cond]
            print(f"{cond:<16}{r['nDCG@5']:>9.4f}{r['nDCG@10']:>9.4f}{r['MRR']:>8.4f}"
                  f"{r['Recall@10']:>11.4f}{r['MAP']:>8.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"config": vars(args), "n_queries": len(queries),
                   "n_docs": len(bills), "results": results}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nSaved -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

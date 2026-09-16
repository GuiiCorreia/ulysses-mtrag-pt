"""Reconstruct the T1 judgment dump of the ICTAI BM25 run — zero API calls.

Rationale (validated in code): at T1 history is empty, so _augment_query returns
the raw query for ALL four strategies (multi_turn_retriever.py:90-91; rewrite's
LLM is never invoked). The T1 pool is therefore the BM25 top-10 of the seed query,
identical across strategies -> 100 convs x 10 docs = 1000 pairs = n_judgments.

Gates (script ABORTS without writing the CSV if any fails):
  G1  every retrieved (query, doc) pair is present in the judgment cache;
  G2  recomputed per-conversation nDCG@10 at T1 matches ndcg_raw_by_turn.T1 of
      mtrag_eval_bm25.json for ALL FOUR strategies, 100/100 each;
  G3  n of agreement pairs == stored n_judgments (1000);
  G4  binary raw agreement matches stored raw_agreement (0.692);
  G5  Fleiss (3-level ordinal, the ORIGINAL compute_fleiss_kappa) matches the
      stored 'kappa' field (0.2584) — confirming that field is Fleiss3, not Cohen.
Additionally reports the NEW binary Cohen kappa (not stored anywhere yet).

Output: results/mtrag_judgments_bm25.csv (T1 rows only; same columns as the
dense dump). T2-T4 pools cannot be reconstructed offline (rewrite is LLM-based).

    uv run python rebuild_bm25_t1_judgments.py
"""
import sys, json, csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mestrado.data.loader import DataLoader
from mestrado.retrieval.bm25 import BM25Retriever
from mestrado.retrieval.multi_turn_retriever import MultiTurnRetriever
from mestrado.synthetic.llm_judge import compute_fleiss_kappa
from mestrado.metrics.ir_metrics import ndcg_at_k

K = 10
STORED = "results/mtrag_eval_bm25.json"
CACHE = "results/cache/judge_mtrag_eval_bm25.json"
CONVS = "results/ulysses_mtrag_100.json"
OUT = "results/mtrag_judgments_bm25.csv"

def main() -> int:
    stored = json.load(open(STORED, encoding="utf-8"))
    cache = json.load(open(CACHE, encoding="utf-8"))
    convs = json.load(open(CONVS, encoding="utf-8"))["conversations"]
    jkey = lambda q, n: f"{q}␟{n}"

    print("Loading Ulysses corpus + queries ...")
    loader = DataLoader()
    bills, queries = loader.load_all()
    qmap = {q.query_id: q for q in queries}
    print(f"  {len(bills):,} bills | {len(queries)} queries")

    print("Building BM25 index (same code/params as the eval run) ...")
    bm25 = BM25Retriever(bills)
    bm25.build_index()
    mtr = MultiTurnRetriever(bm25, strategy="lastturn", provider="deepinfra")

    # ---- retrieve T1 pools ----
    rows = []          # (ci, t1_query, seed_qid, names[10])
    for ci, conv in enumerate(convs):
        t1 = conv["turns"][0]
        assert t1["n"] == 1
        names = [r.bill.name for r in mtr.retrieve(t1["query"], [], k=K)]
        rows.append((ci, t1["query"], str(conv["seed_query_id"]), names))

    # ---- G1: cache coverage ----
    missing = [(ci, n) for ci, q, _, names in rows for n in names
               if jkey(q, n) not in cache]
    print(f"\nG1 cache coverage: {1000 - len(missing)}/1000 pairs in cache "
          f"({'PASS' if not missing else 'FAIL'})")
    if missing:
        print("   missing sample:", missing[:5])
        return 1

    # ---- G2: per-conversation nDCG@10 vs stored, all 4 strategies ----
    print("G2 nDCG@10 T1 vs ndcg_raw_by_turn.T1:")
    g2_fail = False
    recomputed = []
    for ci, q, _, names in rows:
        llm_rel = {n: cache[jkey(q, n)] for n in names}
        recomputed.append(round(ndcg_at_k(names, llm_rel, k=K), 4))
    for strat in ("lastturn", "concat", "questions", "rewrite"):
        stored_t1 = stored["results"][strat]["ndcg_raw_by_turn"]["T1"]
        n_match = sum(1 for a, b in zip(recomputed, stored_t1)
                      if abs(a - b) < 5e-5)
        status = "PASS" if n_match == 100 else "FAIL"
        print(f"   {strat:<10} {n_match}/100 match  [{status}]")
        if n_match != 100:
            g2_fail = True
            diffs = [(i, a, b) for i, (a, b) in enumerate(zip(recomputed, stored_t1))
                     if abs(a - b) >= 5e-5][:5]
            print("     first mismatches (ci, recomputed, stored):", diffs)
    if g2_fail:
        print("ABORT: pools do not reproduce the stored run.")
        return 1

    # ---- agreement: human vs LLM on the validated pools ----
    human, llm = {}, {}
    for ci, q, seed_qid, names in rows:
        qq = qmap.get(seed_qid)
        gr = qq.graded_relevance() if qq else {}
        for n in names:
            key = f"{ci}::{n}"
            human[key] = gr.get(n, 0.0)
            llm[key] = cache[jkey(q, n)]
    items = list(human.keys())
    n_items = len(items)

    raw_bin = sum(1 for k in items if (human[k] >= 0.5) == (llm[k] >= 0.5)) / n_items
    fleiss3 = compute_fleiss_kappa([human, llm], items)

    # new binary Cohen
    po = raw_bin
    ph = sum(1 for k in items if human[k] >= 0.5) / n_items
    pl = sum(1 for k in items if llm[k] >= 0.5) / n_items
    pe = ph * pl + (1 - ph) * (1 - pl)
    cohen_bin = (po - pe) / (1 - pe) if pe < 1 else 0.0
    pabak = 2 * po - 1

    st = stored["t1_human_vs_llm"]
    print(f"\nG3 n pairs: {n_items} vs stored {st['n_judgments']} "
          f"[{'PASS' if n_items == st['n_judgments'] else 'FAIL'}]")
    print(f"G4 raw agreement (binary): {raw_bin:.4f} vs stored {st['raw_agreement']} "
          f"[{'PASS' if abs(raw_bin - st['raw_agreement']) < 5e-4 else 'FAIL'}]")
    print(f"G5 Fleiss 3-level ordinal:  {fleiss3:.4f} vs stored 'kappa' {st['kappa']} "
          f"[{'PASS' if abs(fleiss3 - st['kappa']) < 5e-4 else 'FAIL'}]")
    if (n_items != st["n_judgments"]
            or abs(raw_bin - st["raw_agreement"]) >= 5e-4
            or abs(fleiss3 - st["kappa"]) >= 5e-4):
        print("ABORT: agreement does not reproduce the stored block.")
        return 1

    print(f"\nNEW  Cohen kappa (binary, >=0.5): {cohen_bin:.4f}")
    print(f"NEW  PABAK (binary):              {pabak:.4f}")
    print(f"     marginals: expert+={ph:.4f}  judge+={pl:.4f}")

    # ---- write CSV (T1 only, same columns as the dense dump) ----
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["conv_idx", "turn", "query", "doc_name",
                    "ementa", "text_excerpt", "llm_score"])
        for ci, q, _, names in rows:
            for n in names:
                b = bills.get(n)
                w.writerow([ci, 1, q, n,
                            (b.txt_ementa if b else "")[:600],
                            (b.text if b else "")[:600],
                            cache[jkey(q, n)]])
    print(f"\nALL GATES PASSED -> wrote {OUT} ({n_items} T1 rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

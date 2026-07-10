"""
Multi-turn retrieval evaluation for Ulysses-MTRAG (ICTAI 2026).

Measures how retrieval quality evolves ACROSS TURNS and which context strategy
helps (lastturn / concat / questions / rewrite).

Relevance basis (consistent across turns, so the progression is comparable):
  - Every turn's POOLED retrieved docs (union of strategies, de-duplicated) are
    scored by the LLM judge (API). Unjudged = irrelevant (standard pooling).
  - Turn 1 ALSO has human ground truth (the seed query's Conle judgments), so we
    additionally report human-vs-LLM agreement (Cohen/Fleiss kappa) at T1 — a
    built-in validation of the LLM judge against human labels.

Reuses cached Ulysses BGE-M3 embeddings (results/cache/ulysses/bge-m3.pkl).
You run; the code is written.

    uv run python run_eval_mtrag.py --conversations results/ulysses_mtrag_100.json \
        --provider deepinfra --judge-model "meta-llama/Llama-3.3-70B-Instruct" \
        --rewrite-model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
        --retriever bge-m3 --k 10 --workers 8 --out results/mtrag_eval.json

Tip: add --limit 20 for a cheap smoke test first.
"""
import sys
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mestrado.data.loader import DataLoader
from mestrado.retrieval.bm25 import BM25Retriever
from mestrado.retrieval.dense import DenseRetriever
from mestrado.retrieval.hybrid import HybridRetriever
from mestrado.retrieval.multi_turn_retriever import MultiTurnRetriever, TurnContext
from mestrado.synthetic.llm_judge import get_judge, compute_fleiss_kappa
from mestrado.metrics.ir_metrics import ndcg_at_k, recall_at_k

ULYSSES_CACHE = Path("results/cache/ulysses/bge-m3.pkl")
BGE_M3 = "BAAI/bge-m3"


def main():
    ap = argparse.ArgumentParser(description="Multi-turn retrieval eval (Ulysses-MTRAG)")
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--provider", default="deepinfra", choices=["openrouter", "deepinfra", "ollama"])
    ap.add_argument("--judge-model", default="meta-llama/Llama-3.3-70B-Instruct")
    ap.add_argument("--rewrite-model", default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    help="Small model for query rewriting (rewrite strategy)")
    ap.add_argument("--retriever", default="bge-m3", choices=["bm25", "bge-m3", "hybrid"])
    ap.add_argument("--strategies", nargs="+",
                    default=["lastturn", "concat", "questions", "rewrite"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32,
                    help="Concurrent judge requests (DeepInfra/OpenRouter handle 50+/s)")
    ap.add_argument("--out", default="results/mtrag_eval.json")
    ap.add_argument("--dump-pairs", default=None,
                    help="If set, write per-(query,doc) LLM judgments to this CSV "
                         "(used to sample items for human validation)")
    args = ap.parse_args()

    print(f"\n{'='*68}\nMULTI-TURN RETRIEVAL EVAL (Ulysses-MTRAG)\n{'='*68}")
    print(f"retriever={args.retriever}  strategies={args.strategies}  k={args.k}\n"
          f"judge={args.judge_model} | rewrite={args.rewrite_model} | via {args.provider}")

    # ---- Corpus + queries (Turn-1 human ground truth) ----
    print("\nLoading Ulysses corpus + queries ...")
    loader = DataLoader()
    bills, queries = loader.load_all()
    qmap = {q.query_id: q for q in queries}
    print(f"  {len(bills):,} bills | {len(queries)} queries")

    # ---- Base retriever ----
    print(f"Building base retriever ({args.retriever}) ...")
    bm25 = BM25Retriever(bills); bm25.build_index()
    dense = None
    if args.retriever in ("bge-m3", "hybrid"):
        dense = DenseRetriever(bills, model_name=BGE_M3)
        dense.build_index(cache_path=ULYSSES_CACHE)
    base = {"bm25": bm25, "bge-m3": dense}.get(args.retriever)
    if args.retriever == "hybrid":
        base = HybridRetriever(bm25, dense, bill_key="name")

    retrievers = {
        s: MultiTurnRetriever(base, strategy=s, provider=args.provider,
                              rewrite_model=args.rewrite_model)
        for s in args.strategies
    }

    # ---- Conversations ----
    data = json.load(open(args.conversations, encoding="utf-8"))
    convs = data["conversations"]
    if args.limit:
        convs = convs[:args.limit]
    print(f"  {len(convs)} conversations")

    # ---- Step 1: retrieve for every (conv, turn, strategy) ----
    print("\nRetrieving ...")
    records = []  # (ci, turn_n, seed_qid, query, {strat: [names]}, pooled)
    for ci, conv in enumerate(convs):
        seed_qid = str(conv["seed_query_id"])
        history: list[TurnContext] = []
        for turn in conv["turns"]:
            tn, tq = turn["n"], turn["query"]
            per_strat = {s: [r.bill.name for r in mtr.retrieve(tq, history, k=args.k)]
                         for s, mtr in retrievers.items()}
            pooled = list(dict.fromkeys(n for names in per_strat.values() for n in names))
            records.append((ci, tn, seed_qid, tq, per_strat, pooled))
            history.append(TurnContext(n=tn, query=tq, turn_type=turn.get("type", "unknown")))

    # ---- Step 2: relevance (LLM judge for ALL turns; human GT collected at T1) ----
    # Doc-level parallelism: flatten every (turn, pooled-doc) into ONE job pool so we
    # saturate the API. (judge_batch judges docs sequentially; flattening does not.)
    judge = get_judge(args.provider, model=args.judge_model)

    # Judgment cache (resume + cheap re-runs): key = query|||doc -> score.
    jcache_path = Path("results/cache") / f"judge_{Path(args.out).stem}.json"
    jcache_path.parent.mkdir(parents=True, exist_ok=True)
    jcache = json.load(open(jcache_path, encoding="utf-8")) if jcache_path.exists() else {}
    jkey = lambda q, n: f"{q}␟{n}"

    jobs = []  # (record_index, doc_name, query, ementa, excerpt)
    for ri, (ci, tn, seed_qid, tq, per_strat, pooled) in enumerate(records):
        for n in pooled:
            b = bills.get(n)
            jobs.append((ri, n, tq,
                         b.txt_ementa if b else "",
                         (b.text if b else "")[:500]))

    scores: dict = {}
    todo = []
    for job in jobs:
        ri, name, q = job[0], job[1], job[2]
        k = jkey(q, name)
        if k in jcache:
            scores[(ri, name)] = jcache[k]
        else:
            todo.append(job)
    print(f"Judging {len(todo)} (query,doc) pairs ({len(jobs)-len(todo)} cached) "
          f"over {len(records)} turns with {args.workers} workers ...")

    def judge_one(job):
        ri, name, q, ementa, excerpt = job
        return ri, name, q, judge.judge_single(q, name, ementa, excerpt).score

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(judge_one, j) for j in todo]
        done = 0
        for f in as_completed(futs):
            ri, name, q, sc = f.result()
            scores[(ri, name)] = sc
            jcache[jkey(q, name)] = sc
            done += 1
            if done % 500 == 0 or done == len(todo):
                json.dump(jcache, open(jcache_path, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"  {done}/{len(todo)} (cache saved)")

    scored = []
    for ri, (ci, tn, seed_qid, tq, per_strat, pooled) in enumerate(records):
        llm_rel = {n: scores.get((ri, n), 0.0) for n in pooled}
        human_rel = None
        if tn == 1:
            q = qmap.get(seed_qid)
            gr = q.graded_relevance() if q else {}
            human_rel = {n: gr.get(n, 0.0) for n in pooled}
        scored.append((ci, tn, tq, pooled, per_strat, llm_rel, human_rel))

    # ---- Step 3: metrics (consistent LLM basis) + T1 human-vs-LLM kappa ----
    ndcg = defaultdict(lambda: defaultdict(list))
    rec  = defaultdict(lambda: defaultdict(list))
    human_T1, llm_T1 = {}, {}
    for ci, tn, tq, pooled, per_strat, llm_rel, human_rel in scored:
        for s, names in per_strat.items():
            ndcg[s][tn].append(ndcg_at_k(names, llm_rel, k=args.k))
            rec[s][tn].append(recall_at_k(names, llm_rel, k=args.k))
        if tn == 1 and human_rel is not None:
            for n, hs in human_rel.items():
                key = f"{ci}::{n}"
                human_T1[key] = hs
                llm_T1[key] = llm_rel.get(n, 0.0)

    avg = lambda l: round(sum(l) / len(l), 4) if l else 0.0
    turns = sorted({tn for s in ndcg for tn in ndcg[s]})

    # T1 LLM-judge vs human (Conle) agreement — validates the judge
    t1_items = list(human_T1.keys())
    t1_kappa = round(compute_fleiss_kappa([human_T1, llm_T1], t1_items), 4) if t1_items else None
    t1_agree = (round(sum(1 for k in t1_items
                          if (human_T1[k] >= 0.5) == (llm_T1[k] >= 0.5)) / len(t1_items), 4)
                if t1_items else None)

    print(f"\n{'='*68}\nnDCG@{args.k}  by strategy x turn (LLM-judged, consistent)\n{'='*68}")
    head = "strategy".ljust(12) + "".join(f"T{t}".rjust(9) for t in turns) + "mean".rjust(9)
    print(head); print("-" * len(head))
    results = {}
    for s in args.strategies:
        per_turn = [avg(ndcg[s][t]) for t in turns]
        overall = avg([x for t in turns for x in ndcg[s][t]])
        print(s.ljust(12) + "".join(f"{v:9.4f}" for v in per_turn) + f"{overall:9.4f}")
        results[s] = {
            "ndcg_by_turn":   {f"T{t}": avg(ndcg[s][t]) for t in turns},
            "recall_by_turn": {f"T{t}": avg(rec[s][t]) for t in turns},
            "ndcg_overall":   overall,
            # Per-conversation raw nDCG per turn -> enables CIs + paired significance tests.
            "ndcg_raw_by_turn": {f"T{t}": [round(x, 4) for x in ndcg[s][t]] for t in turns},
        }
    print(f"\nT1 human(Conle)-vs-LLM judge:  Cohen/Fleiss kappa={t1_kappa}  "
          f"raw_agreement={t1_agree}  (n={len(t1_items)} doc judgments)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(args), "n_conversations": len(convs),
               "t1_human_vs_llm": {"kappa": t1_kappa, "raw_agreement": t1_agree,
                                   "n_judgments": len(t1_items)},
               "results": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved -> {args.out}")

    if args.dump_pairs:
        import csv
        with open(args.dump_pairs, "w", newline="", encoding="utf-8") as fcsv:
            w = csv.writer(fcsv)
            w.writerow(["conv_idx", "turn", "query", "doc_name",
                        "ementa", "text_excerpt", "llm_score"])
            for ci, tn, tq, pooled, per_strat, llm_rel, human_rel in scored:
                for n in pooled:
                    b = bills.get(n)
                    w.writerow([ci, tn, tq, n,
                                (b.txt_ementa if b else "")[:600],
                                (b.text if b else "")[:600],
                                llm_rel.get(n, 0.0)])
        print(f"Dumped per-pair judgments -> {args.dump_pairs}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

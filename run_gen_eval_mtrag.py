"""
Multi-turn GENERATION evaluation for Ulysses-MTRAG (ICTAI 2026).

Answers the question MTRAG/SemEval Task B/C ask: does retrieval degradation
propagate to the generated answer? For each turn we retrieve context, generate an
answer with each SLM, and score it on three axes (following MTRAG's 3-metric
generation evaluation):
  - FAITH  : reference-less faithfulness to retrieved context (RAGAS-style, LLM judge)
  - SIM    : reference similarity to an Oracle answer (BERTScore-F1, ROUGE-L)  [RB_alg proxy]
  - FANC   : LLM-judge faithfulness/appropriateness/completeness vs Oracle ref  [RB_llm proxy]

Retrieval = current-turn query (last-turn), top-5; generation prompt includes the
conversation history so the model can resolve references. Doc/turn-level parallelism.

Run from codigos_mestrado/ (smoke first):
    uv run python run_gen_eval_mtrag.py --conversations results/ulysses_mtrag_100.json \
        --limit 10 --workers 32 --out results/mtrag_gen_smoke.json
Full:
    uv run python run_gen_eval_mtrag.py --conversations results/ulysses_mtrag_100.json \
        --workers 32 --out results/mtrag_gen.json
"""
import sys, json, argparse, time, re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mestrado.data.loader import DataLoader
from mestrado.retrieval.bm25 import BM25Retriever
from mestrado.retrieval.dense import DenseRetriever
from mestrado.generation.rag import DeepInfraRAG, OpenRouterRAG
from mestrado.reranking.llm_reranker import RerankerResult
from mestrado.metrics.generation_metrics import evaluate_generation

ULYSSES_CACHE = Path("results/cache/ulysses/bge-m3.pkl")
BGE_M3 = "BAAI/bge-m3"

# Best performers from the single-turn (ENIAC) study: the 7-8B faithfulness
# plateau + the 31B upper bound. Slugs/providers verified vs run_generation_2x2.
SLM_MODELS = [
    "microsoft/phi-4",              # 14B   (DeepInfra)
    "qwen/qwen-2.5-7b-instruct",     # 7B    (OpenRouter)
    "qwen/qwen3-8b",                 # 8B    (OpenRouter)
    "mistralai/ministral-8b-2512",   # 8B    (OpenRouter)
    "google/gemma-4-31B-it",         # 31B upper bound (DeepInfra)
]
# Oracle (reference answers) on DeepInfra — matches the ENIAC 2x2 setup
# (run_generation_2x2: Qwen/Qwen2.5-72B-Instruct, deepinfra) and avoids the
# OpenRouter shared-pool rate limits / Novita "no completions endpoint" failures
# that hit qwen-2.5-72b when routed through OpenRouter.
ORACLE_MODEL = "Qwen/Qwen2.5-72B-Instruct"
JUDGE_MODEL  = "meta-llama/Llama-3.3-70B-Instruct"
RETRY = [0, 5, 15]

# Slugs forced to DeepInfra even though a hint below would otherwise route them
# to OpenRouter (the DeepInfra Qwen slug also starts with "qwen/").
_DEEPINFRA_FORCE = {"Qwen/Qwen2.5-72B-Instruct"}
_OPENROUTER_HINTS = ("qwen/", "deepseek/", "ministral")


def get_rag(model):
    if model in _DEEPINFRA_FORCE:
        return DeepInfraRAG(model=model)
    m = model.lower()
    if any(h in m for h in _OPENROUTER_HINTS):
        return OpenRouterRAG(model=model)
    return DeepInfraRAG(model=model)


def gen_with_retry(rag, query, reranked):
    for d in RETRY:
        if d:
            time.sleep(d)
        try:
            r = rag.generate(query, reranked, top_bills=5)
            if r and r.tokens_generated and not str(r.response).startswith("[Erro"):
                return r.response
        except Exception as e:
            last = e
    return ""


def judge_score(client, model, prompt):
    """One judge call returning a float 0-1 (robust parse)."""
    try:
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": prompt}],
            temperature=0, max_tokens=20)
        m = re.search(r"(0?\.\d+|1\.0|0|1)", resp.choices[0].message.content.strip())
        return max(0.0, min(1.0, float(m.group(1)))) if m else 0.5
    except Exception:
        return 0.5


def faith_prompt(answer, context):
    return (f"Avalie a FIDELIDADE da resposta ao contexto (0.0-1.0): 1.0=tudo vem do "
            f"contexto, 0.0=inventou.\n\nContexto:\n{context[:1500]}\n\nResposta:\n{answer[:800]}\n\nScore:")


def fanc_prompt(answer, reference):
    return (f"Compare a RESPOSTA a uma resposta de REFERÊNCIA quanto a fidelidade, "
            f"adequação e completude. Retorne um único número 0.0-1.0 "
            f"(1.0=equivalente à referência).\n\nREFERÊNCIA:\n{reference[:800]}\n\n"
            f"RESPOSTA:\n{answer[:800]}\n\nScore:")


def main():
    ap = argparse.ArgumentParser(description="Multi-turn generation eval (Ulysses-MTRAG)")
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--models", nargs="+", default=SLM_MODELS)
    ap.add_argument("--oracle", default=ORACLE_MODEL)
    ap.add_argument("--judge", default=JUDGE_MODEL)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="results/mtrag_gen.json")
    ap.add_argument("--score-oracle", action="store_true",
                    help="Also judge the oracle's own answers on the reference-less "
                         "faithfulness axis (within-family scale anchor for the paper)")
    args = ap.parse_args()

    print(f"\n{'='*68}\nMULTI-TURN GENERATION EVAL (Ulysses-MTRAG)\n{'='*68}")
    print(f"models={len(args.models)} | oracle={args.oracle} | judge={args.judge} | workers={args.workers}")

    loader = DataLoader(); bills, _ = loader.load_all()
    print(f"  {len(bills):,} bills")
    dense = DenseRetriever(bills, model_name=BGE_M3); dense.build_index(cache_path=ULYSSES_CACHE)

    convs = json.load(open(args.conversations, encoding="utf-8"))["conversations"]
    if args.limit:
        convs = convs[:args.limit]

    # ---- retrieve context + build history-aware gen query per turn ----
    print("Retrieving context per turn ...")
    units = []  # (ci, tn, gen_query, context_str)
    for ci, c in enumerate(convs):
        hist = []
        for t in c["turns"]:
            tn, tq = t["n"], t["query"]
            res = dense.retrieve(tq, top_k=5)
            ctx = "\n\n---\n\n".join(f"### {r.bill.name}\n{r.bill.txt_ementa}" for r in res)
            reranked = [RerankerResult(bill=r.bill, rank=i + 1) for i, r in enumerate(res)]
            hpfx = ("Histórico da conversa:\n" + "\n".join(f"  T{h['n']}: {h['query']}" for h in hist) + "\n\n") if hist else ""
            units.append((ci, tn, hpfx + tq, ctx, reranked))
            hist.append(t)

    all_models = [args.oracle] + list(args.models)
    rags = {m: get_rag(m) for m in all_models}

    # ---- generate: cached + flattened for parallelism + RESUME ----
    # Cache keyed by (conv, turn, model); only missing/empty answers are (re)generated,
    # and the cache is flushed periodically so a crash/blip never loses prior work.
    cache_path = Path("results/cache") / f"gen_{Path(args.out).stem}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}
    ckey = lambda ci, tn, m: f"{ci}|{tn}|{m}"

    gen, todo = {}, []
    for ui in range(len(units)):
        ci, tn = units[ui][0], units[ui][1]
        for m in all_models:
            cached = cache.get(ckey(ci, tn, m), "")
            if cached:
                gen[(ui, m)] = cached          # reuse successful prior generation
            else:
                todo.append((ui, m))           # missing or previously-failed -> (re)generate
    print(f"Generating {len(todo)} answers "
          f"({len(units)*len(all_models) - len(todo)} reused from cache), "
          f"{args.workers} workers ...")

    def do_gen(job):
        ui, model = job
        ci, tn, q, ctx, reranked = units[ui]
        return ui, model, gen_with_retry(rags[model], q, reranked)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(do_gen, j) for j in todo]
        done = 0
        for f in as_completed(futs):
            ui, model, resp = f.result()
            gen[(ui, model)] = resp
            if resp:  # only cache successful answers; empties stay re-tryable
                cache[ckey(units[ui][0], units[ui][1], model)] = resp
            done += 1
            if done % 200 == 0 or done == len(todo):
                json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"  gen {done}/{len(todo)} (cache saved -> {cache_path.name})")

    # ---- judge FAITH (reference-less) + FANC (vs oracle), parallel + cached ----
    judge = get_rag(args.judge)._client
    jcache_path = Path("results/cache") / f"judge_{Path(args.out).stem}.json"
    jcache = json.load(open(jcache_path, encoding="utf-8")) if jcache_path.exists() else {}
    jkey = lambda ci, tn, m: f"{ci}|{tn}|{m}"
    faith, fanc = {}, {}

    def do_judge(job):
        ui, model = job
        ci, tn, q, ctx, reranked = units[ui]
        ans = gen[(ui, model)]
        fa = judge_score(judge, args.judge, faith_prompt(ans, ctx))
        if model == args.oracle:  # oracle: faithfulness only (no self-reference FANC)
            return ui, model, fa, 0.0
        ref = gen[(ui, args.oracle)]
        return ui, model, fa, judge_score(judge, args.judge, fanc_prompt(ans, ref))

    judged_models = list(args.models) + ([args.oracle] if args.score_oracle else [])
    sjobs = []
    for ui in range(len(units)):
        ci, tn = units[ui][0], units[ui][1]
        for m in judged_models:
            c = jcache.get(jkey(ci, tn, m))
            if c:
                faith[(ui, m)], fanc[(ui, m)] = c[0], c[1]
            else:
                sjobs.append((ui, m))
    print(f"Judging {len(sjobs)} pairs "
          f"({len(units)*len(judged_models) - len(sjobs)} cached), {args.workers} workers ...")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(do_judge, j) for j in sjobs]
        done = 0
        for f in as_completed(futs):
            ui, model, fa, fn = f.result()
            faith[(ui, model)] = fa; fanc[(ui, model)] = fn
            jcache[jkey(units[ui][0], units[ui][1], model)] = [fa, fn]
            done += 1
            if done % 500 == 0 or done == len(sjobs):
                json.dump(jcache, open(jcache_path, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"  judge {done}/{len(sjobs)} (cache saved)")

    # ---- reference similarity (BERTScore-F1 + ROUGE-L vs oracle), batched per model ----
    print("Computing reference similarity (BERTScore/ROUGE vs Oracle) ...")
    sim = {}  # model -> (bert_f1_scores, rougeL_scores) aligned to units order
    refs = [gen[(ui, args.oracle)] for ui in range(len(units))]
    for m in args.models:
        preds = [gen[(ui, m)] for ui in range(len(units))]
        gm = evaluate_generation(preds, refs, lang="pt")
        sim[m] = (gm.bertscore_f1_scores, gm.rouge_l_scores)

    # ---- aggregate per model (overall + by turn) ----
    avg = lambda l: round(sum(l) / len(l), 4) if l else 0.0
    results = {}
    for m in args.models:
        bert, rouge = sim[m]
        by_turn = defaultdict(lambda: {"faith": [], "fanc": [], "bert": [], "rouge": []})
        F, N, B, R = [], [], [], []
        per_unit = []  # per (conv,turn) raw scores -> enables CIs + paired significance tests
        for ui in range(len(units)):
            ci, tn, q, ctx, reranked = units[ui]
            by_turn[tn]["faith"].append(faith[(ui, m)]); by_turn[tn]["fanc"].append(fanc[(ui, m)])
            by_turn[tn]["bert"].append(bert[ui]); by_turn[tn]["rouge"].append(rouge[ui])
            F.append(faith[(ui, m)]); N.append(fanc[(ui, m)]); B.append(bert[ui]); R.append(rouge[ui])
            per_unit.append({"ci": ci, "tn": tn, "faith": round(faith[(ui, m)], 4),
                             "bert": round(bert[ui], 4), "rouge": round(rouge[ui], 4)})
        results[m] = {
            "faithfulness": avg(F), "fanc": avg(N), "bertscore_f1": avg(B), "rouge_l": avg(R),
            "by_turn": {f"T{t}": {"faith": avg(by_turn[t]["faith"]), "fanc": avg(by_turn[t]["fanc"]),
                                  "bert": avg(by_turn[t]["bert"]), "rouge": avg(by_turn[t]["rouge"])}
                        for t in sorted(by_turn)},
            "per_unit": per_unit,
        }

    if args.score_oracle:
        of = [faith[(ui, args.oracle)] for ui in range(len(units))]
        obt = defaultdict(list)
        for ui in range(len(units)):
            obt[units[ui][1]].append(faith[(ui, args.oracle)])
        results["__oracle__"] = {
            "model": args.oracle, "faithfulness": avg(of),
            "faith_by_turn": {f"T{t}": avg(v) for t, v in sorted(obt.items())},
        }
        print(f"\nORACLE ({args.oracle}) reference-less faithfulness: {avg(of):.3f}  "
              + "  ".join(f"T{t}={avg(v):.3f}" for t, v in sorted(obt.items())))

    print(f"\n{'='*68}\nGENERATION QUALITY by model\n{'='*68}")
    print(f"{'model':<34}{'Faith':>8}{'FANC':>8}{'BERT':>8}{'ROUGE-L':>9}")
    print("-" * 67)
    for m in sorted(args.models, key=lambda x: results[x]["faithfulness"], reverse=True):
        r = results[m]
        print(f"{m.split('/')[-1][:32]:<34}{r['faithfulness']:>8.3f}{r['fanc']:>8.3f}"
              f"{r['bertscore_f1']:>8.3f}{r['rouge_l']:>9.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": {k: v for k, v in vars(args).items()},
               "n_conversations": len(convs), "n_turns": len(units), "results": results},
              open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

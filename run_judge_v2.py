"""Faithfulness / FANC judging — PROTOCOL v2 (ICTAI camera-ready; decision of 2026-09-05).

Why: the original judge call used max_tokens=20 and mapped any reply without a number
to 0.5 (judge_score() in run_gen_eval_mtrag.py). The judge often opens with a sentence
("Avaliando a fidelidade da resposta ...") that consumes the 20 tokens, so 21-27% of
the stored scores are silent parse fallbacks, not judgments.

Protocol v2 (do not change without notice):
  (1) SAME generations: results/cache/gen_mtrag_gen.json is only read (sha256 recorded);
      400 units x (5 SLMs + oracle) = 2,400 faithfulness judgments; FANC for 400 x 5 SLMs.
  (2) SAME judge (JUDGE_MODEL via the same provider), temperature 0, SAME faith_prompt /
      fanc_prompt text plus ONE final line: "Responda apenas com um número entre 0.0 e 1.0.";
      max_tokens=64.
  (3) Parser: first number in the form 0.x / 1.0 / 0 / 1 (decimal comma accepted);
      no number -> NA (None). NEVER 0.5. API exception -> up to 3 attempts with backoff,
      then NA.
  (4) New cache per round: results/cache/judge_faith_v2_r<N>.json with, per "ci|tn|model",
      "faith": [score, raw_reply, attempts] and "fanc": [score, raw_reply, attempts]
      (+ "faith_usage"/"fanc_usage": [prompt_tokens, completion_tokens] for the cost line).
      Output: results/mtrag_gen_v2_r<N>.json (same layout as mtrag_gen.json + n_na fields).
      Nothing pre-existing is overwritten.
  (5) --estimate judges 20 units first and projects the round cost from REAL prompt tokens.
  (6) Complete-case companion file (units with non-NA faith for every judged model, same
      order) for analyze_significance_mtrag.py / analyze_scale.py, which pair by position.

Contexts (top-5 dense retrieval per turn, exactly as run_gen_eval_mtrag.py builds them) are
materialized once in results/cache/contexts_v2.json so both rounds see byte-identical prompts.

    uv run python run_judge_v2.py --round 1 --estimate
    uv run python run_judge_v2.py --round 1
    uv run python run_judge_v2.py --round 2 --sim-from results/mtrag_gen_v2_r1.json
"""
import sys, os, re, json, time, hashlib, argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).parent / "src"))
from run_gen_eval_mtrag import (get_rag, faith_prompt, fanc_prompt, SLM_MODELS, ORACLE_MODEL,
                                JUDGE_MODEL, BGE_M3, ULYSSES_CACHE)
from mestrado.metrics.generation_metrics import evaluate_generation

SUFFIX_LINE = "\nResponda apenas com um número entre 0.0 e 1.0."
MAX_TOKENS = 64
BACKOFF = [0, 2, 6]                     # seconds before attempt 1, 2, 3
GEN_CACHE = Path("results/cache/gen_mtrag_gen.json")
CONTEXTS = Path("results/cache/contexts_v2.json")
CONVS = "results/ulysses_mtrag_100.json"
PRICES = {"turbo_0.10_0.32": (0.10, 0.32), "conservative_0.23_0.40": (0.23, 0.40)}

# first number in 0.x / 1.0 / 0 / 1 form; decimal comma accepted; a sentence-final "." after the
# number is allowed (not followed by a digit); "10", "100%", "1.5" are rejected.
_NUM = re.compile(r"(?<![\d.,])(1[.,]0+|0[.,]\d+|[01])(?![\d,%]|[.,]\d)")


def parse_score(text):
    """First number in 0.x / 1.0 / 0 / 1 form (decimal comma ok); None if absent. Never 0.5 by default."""
    if not text:
        return None
    m = _NUM.search(text)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return max(0.0, min(1.0, v))


def judge_call(client, prompt):
    """Returns (score|None, raw_reply, attempts, prompt_tokens, completion_tokens)."""
    raw, tin, tout = "", 0, 0
    for attempt, delay in enumerate(BACKOFF, start=1):
        if delay:
            time.sleep(delay)
        try:
            resp = client.chat.completions.create(
                model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}],
                temperature=0, max_tokens=MAX_TOKENS)
            raw = (resp.choices[0].message.content or "").strip()
            u = getattr(resp, "usage", None)
            tin, tout = (getattr(u, "prompt_tokens", 0) or 0), (getattr(u, "completion_tokens", 0) or 0)
            return parse_score(raw), raw, attempt, tin, tout
        except Exception as e:
            raw = f"ERROR: {type(e).__name__}: {str(e)[:160]}"
    return None, raw, len(BACKOFF), tin, tout


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_or_load_contexts():
    if CONTEXTS.exists():
        d = json.load(open(CONTEXTS, encoding="utf-8"))
        print(f"contexts: loaded {len(d['units'])} units from {CONTEXTS} (device={d['device']})")
        return d["units"]
    from mestrado.data.loader import DataLoader
    from mestrado.retrieval.dense import DenseRetriever, _DEVICE
    convs = json.load(open(CONVS, encoding="utf-8"))["conversations"]
    print(f"contexts: building on device={_DEVICE} (loading corpus + cached BGE-M3 index) ...")
    bills, _ = DataLoader().load_all()
    dense = DenseRetriever(bills, model_name=BGE_M3); dense.build_index(cache_path=ULYSSES_CACHE)
    units = []
    for ci, c in enumerate(convs):
        for t in c["turns"]:
            res = dense.retrieve(t["query"], top_k=5)
            ctx = "\n\n---\n\n".join(f"### {r.bill.name}\n{r.bill.txt_ementa}" for r in res)
            units.append({"ci": ci, "tn": t["n"], "top5": [r.bill.name for r in res], "ctx": ctx})
    CONTEXTS.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"device": _DEVICE, "k": 5, "model": BGE_M3, "index_cache": str(ULYSSES_CACHE),
               "units": units}, open(CONTEXTS, "w", encoding="utf-8"), ensure_ascii=False)
    print(f"contexts: wrote {len(units)} units -> {CONTEXTS}")
    return units


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", type=int, required=True, choices=[1, 2])
    ap.add_argument("--estimate", action="store_true", help="judge 20 units, project cost, exit")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--sim-from", default=None, help="reuse per-unit BERTScore/ROUGE from this v2 output")
    ap.add_argument("--original", default="results/mtrag_gen.json", help="original round (to verify bert/rouge)")
    args = ap.parse_args()

    cache_path = Path(f"results/cache/judge_faith_v2_r{args.round}.json")
    out_path = Path(f"results/mtrag_gen_v2_r{args.round}.json")
    cc_path = Path(f"results/mtrag_gen_v2_r{args.round}.complete.json")
    if out_path.exists() and not args.estimate:
        print(f"REFUSING to overwrite existing {out_path}"); return 1

    gen_sha = sha256(GEN_CACHE)
    gen = json.load(open(GEN_CACHE, encoding="utf-8"))
    print(f"gen cache: {GEN_CACHE} sha256={gen_sha} bytes={GEN_CACHE.stat().st_size} entries={len(gen)} (read-only)")

    units = build_or_load_contexts()
    all_models = list(SLM_MODELS) + [ORACLE_MODEL]
    ckey = lambda ci, tn, m: f"{ci}|{tn}|{m}"

    cache = json.load(open(cache_path, encoding="utf-8")) if cache_path.exists() else {}
    client = get_rag(JUDGE_MODEL)._client

    # ---- jobs: (ui, model, kind)
    jobs = []
    for ui, u in enumerate(units):
        for m in all_models:
            ent = cache.get(ckey(u["ci"], u["tn"], m), {})
            if "faith" not in ent:
                jobs.append((ui, m, "faith"))
            if m != ORACLE_MODEL and "fanc" not in ent:
                jobs.append((ui, m, "fanc"))
    if args.estimate:
        # first 20 units, SLMs only: faith + fanc (= 40 calls) — cached as valid v2 judgments
        jobs = [j for j in jobs if j[0] < 20 and j[1] != ORACLE_MODEL][:40]
    print(f"round {args.round}: {len(jobs)} judge calls to make "
          f"({sum(1 for j in jobs if j[2]=='faith')} faith, {sum(1 for j in jobs if j[2]=='fanc')} fanc), "
          f"workers={args.workers}, max_tokens={MAX_TOKENS}")

    def do(job):
        ui, m, kind = job
        u = units[ui]
        ans = gen.get(ckey(u["ci"], u["tn"], m), "")
        if kind == "faith":
            prompt = faith_prompt(ans, u["ctx"]) + SUFFIX_LINE
        else:
            ref = gen.get(ckey(u["ci"], u["tn"], ORACLE_MODEL), "")
            prompt = fanc_prompt(ans, ref) + SUFFIX_LINE
        return job, judge_call(client, prompt)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(do, j) for j in jobs]):
            (ui, m, kind), (score, raw, attempts, tin, tout) = fut.result()
            u = units[ui]
            ent = cache.setdefault(ckey(u["ci"], u["tn"], m), {})
            ent[kind] = [score, raw, attempts]
            ent[kind + "_usage"] = [tin, tout]
            done += 1
            if done % 400 == 0 or done == len(jobs):
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                json.dump(cache, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
                print(f"  judged {done}/{len(jobs)} (cache saved -> {cache_path.name})")

    # ---- cost from REAL usage
    def usage_stats(kind):
        us = [v[kind + "_usage"] for v in cache.values() if kind + "_usage" in v]
        n = len(us); tin = sum(x[0] for x in us); tout = sum(x[1] for x in us)
        return n, tin, tout
    nf, fin, fout = usage_stats("faith"); nn, nin, nout = usage_stats("fanc")
    if args.estimate:
        mf_in, mf_out = fin / max(1, nf), fout / max(1, nf)
        mn_in, mn_out = nin / max(1, nn), nout / max(1, nn)
        tot_in = 2400 * mf_in + 2000 * mn_in; tot_out = 2400 * mf_out + 2000 * mn_out
        print(f"\nESTIMATE from real usage: faith calls n={nf} mean tokens in/out={mf_in:.0f}/{mf_out:.1f}; "
              f"fanc n={nn} mean in/out={mn_in:.0f}/{mn_out:.1f}")
        print(f"  projected round: 2400 faith + 2000 fanc = 4400 calls, ~{tot_in/1e6:.2f} M prompt tokens, "
              f"~{tot_out/1e3:.0f} k completion tokens")
        for name, (pi, po) in PRICES.items():
            print(f"  cost @ {name}: US$ {tot_in/1e6*pi + tot_out/1e6*po:.2f}")
        na = sum(1 for v in cache.values() for k in ("faith", "fanc") if k in v and v[k][0] is None)
        print(f"  NA so far: {na} of {nf+nn}; sample raw replies: "
              f"{[v['faith'][1][:30] for v in list(cache.values())[:4] if 'faith' in v]}")
        return 0

    # ---- reference similarity (BERTScore / ROUGE): reuse or recompute, then verify vs original
    orig = json.load(open(args.original, encoding="utf-8"))["results"] if Path(args.original).exists() else {}
    sim = {}
    if args.sim_from:
        src = json.load(open(args.sim_from, encoding="utf-8"))["results"]
        for m in SLM_MODELS:
            pu = {(x["ci"], x["tn"]): (x["bert"], x["rouge"]) for x in src[m]["per_unit"]}
            sim[m] = ([pu[(u["ci"], u["tn"])][0] for u in units], [pu[(u["ci"], u["tn"])][1] for u in units])
        print(f"bert/rouge reused from {args.sim_from}")
    else:
        print("Computing reference similarity (BERTScore/ROUGE vs oracle) ...")
        refs = [gen.get(ckey(u["ci"], u["tn"], ORACLE_MODEL), "") for u in units]
        for m in SLM_MODELS:
            preds = [gen.get(ckey(u["ci"], u["tn"], m), "") for u in units]
            gm = evaluate_generation(preds, refs, lang="pt")
            sim[m] = (gm.bertscore_f1_scores, gm.rouge_l_scores)
    if orig:
        for m in SLM_MODELS:
            pu = {(x["ci"], x["tn"]): (x["bert"], x["rouge"]) for x in orig[m]["per_unit"]}
            db = max(abs(round(sim[m][0][i], 4) - pu[(u["ci"], u["tn"])][0]) for i, u in enumerate(units))
            dr = max(abs(round(sim[m][1][i], 4) - pu[(u["ci"], u["tn"])][1]) for i, u in enumerate(units))
            print(f"  verify vs original per-unit: {m.split('/')[-1][:22]:<24} max|dBERT|={db:.4f} max|dROUGE|={dr:.4f}")

    # ---- aggregate
    def mean_nn(vals):
        v = [x for x in vals if x is not None]
        return (round(sum(v) / len(v), 4) if v else None), (len(vals) - len(v))
    results = {}
    for m in SLM_MODELS:
        bert, rouge = sim[m]
        per_unit, F, N = [], [], []
        by_turn = defaultdict(lambda: {"faith": [], "fanc": [], "bert": [], "rouge": []})
        for i, u in enumerate(units):
            ent = cache[ckey(u["ci"], u["tn"], m)]
            fa, fn = ent["faith"][0], ent["fanc"][0]
            F.append(fa); N.append(fn)
            t = u["tn"]
            by_turn[t]["faith"].append(fa); by_turn[t]["fanc"].append(fn)
            by_turn[t]["bert"].append(bert[i]); by_turn[t]["rouge"].append(rouge[i])
            per_unit.append({"ci": u["ci"], "tn": t, "faith": None if fa is None else round(fa, 4),
                             "fanc": None if fn is None else round(fn, 4),
                             "bert": round(bert[i], 4), "rouge": round(rouge[i], 4)})
        faith_m, na_f = mean_nn(F); fanc_m, na_n = mean_nn(N)
        bt = {}
        for t in sorted(by_turn):
            fm, nf_ = mean_nn(by_turn[t]["faith"]); nm, nn_ = mean_nn(by_turn[t]["fanc"])
            bt[f"T{t}"] = {"faith": fm, "fanc": nm, "bert": round(sum(by_turn[t]["bert"]) / len(by_turn[t]["bert"]), 4),
                           "rouge": round(sum(by_turn[t]["rouge"]) / len(by_turn[t]["rouge"]), 4),
                           "n_na_faith": nf_, "n_na_fanc": nn_}
        results[m] = {"faithfulness": faith_m, "fanc": fanc_m,
                      "bertscore_f1": round(sum(bert) / len(bert), 4), "rouge_l": round(sum(rouge) / len(rouge), 4),
                      "n_na": {"faith": na_f, "fanc": na_n}, "by_turn": bt, "per_unit": per_unit}
    # oracle
    of, obt, opu = [], defaultdict(list), []
    for u in units:
        fa = cache[ckey(u["ci"], u["tn"], ORACLE_MODEL)]["faith"][0]
        of.append(fa); obt[u["tn"]].append(fa)
        opu.append({"ci": u["ci"], "tn": u["tn"], "faith": None if fa is None else round(fa, 4)})
    om, ona = mean_nn(of)
    results["__oracle__"] = {"model": ORACLE_MODEL, "faithfulness": om,
                             "faith_by_turn": {f"T{t}": mean_nn(v)[0] for t, v in sorted(obt.items())},
                             "n_na": ona, "n_na_by_turn": {f"T{t}": mean_nn(v)[1] for t, v in sorted(obt.items())},
                             "per_unit": opu}

    cost = {"calls_faith": nf, "calls_fanc": nn, "prompt_tokens": fin + nin, "completion_tokens": fout + nout,
            **{f"usd_{k}": round((fin + nin) / 1e6 * p[0] + (fout + nout) / 1e6 * p[1], 3) for k, p in PRICES.items()}}
    out = {"protocol": {"version": "v2", "round": args.round, "judge": JUDGE_MODEL, "temperature": 0,
                        "max_tokens": MAX_TOKENS, "prompt_suffix": SUFFIX_LINE.strip(), "parser": "first 0.x/1.0/0/1, NA if none",
                        "retries": len(BACKOFF), "backoff_s": BACKOFF, "contexts": str(CONTEXTS)},
           "gen_cache_sha256": gen_sha, "cost": cost, "n_conversations": 100, "results": results}
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved -> {out_path}")

    # ---- complete-case companion (units with non-NA faith for every judged model), same order
    judged = list(SLM_MODELS) + ["__oracle__"]
    keep = []
    for i, u in enumerate(units):
        ok = all(results[m]["per_unit"][i]["faith"] is not None for m in judged)
        keep.append(ok)
    cc = json.loads(json.dumps(out))
    for m in judged:
        pu = [x for x, k in zip(cc["results"][m]["per_unit"], keep) if k]
        cc["results"][m]["per_unit"] = pu
        vals = [x["faith"] for x in pu]
        cc["results"][m]["faithfulness"] = round(sum(vals) / len(vals), 4)
        if m != "__oracle__":
            cc["results"][m]["fanc"] = None  # not used by the paired-test scripts
    cc["complete_cases"] = {"n_units": sum(keep), "of": len(units)}
    json.dump(cc, open(cc_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"Complete cases: {sum(keep)}/{len(units)} units -> {cc_path}")

    print("\nTABLE IV (v2, round %d) — faith mean over non-NA | n_na | FANC | BERT | ROUGE-L" % args.round)
    for m in sorted(SLM_MODELS, key=lambda x: -(results[x]["faithfulness"] or 0)):
        r = results[m]
        print(f"  {m.split('/')[-1][:24]:<26} {r['faithfulness']:.3f}  na={r['n_na']['faith']:>3}  "
              f"{r['fanc']:.3f} (na={r['n_na']['fanc']})  {r['bertscore_f1']:.3f}  {r['rouge_l']:.3f}")
    print(f"  {'oracle (' + ORACLE_MODEL.split('/')[-1][:14] + ')':<26} {om:.3f}  na={ona:>3}")
    print(f"cost (real usage): {cost}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

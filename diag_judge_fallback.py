"""Are the exact-0.5 faithfulness scores genuine judgments or silent error fallbacks?

judge_score() in run_gen_eval_mtrag.py returns 0.5 on ANY exception (rate limit,
timeout, parse failure). Round 1 has 26.7% and round 2 21.1% of units at exactly
0.5, so this matters for the absolute means and for the round-to-round shift.

This diagnostic re-judges a sample of units whose round-2 score is exactly 0.5,
SEQUENTIALLY (no concurrency -> no rate limiting), with explicit error capture
and the raw judge reply recorded. Nothing in results/ is modified.

    uv run python diag_judge_fallback.py --n 60
"""
import sys, json, argparse, random, re, time
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from run_gen_eval_mtrag import get_rag, faith_prompt, BGE_M3, ULYSSES_CACHE, JUDGE_MODEL
from mestrado.data.loader import DataLoader
from mestrado.retrieval.dense import DenseRetriever

random.seed(42)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--round", default="results/cache/judge_mtrag_gen_round2.json")
    ap.add_argument("--gen-cache", default="results/cache/gen_mtrag_gen.json")
    ap.add_argument("--out", default="results/diag_judge_fallback.json")
    args = ap.parse_args()

    jc = json.load(open(args.round, encoding="utf-8"))
    gc = json.load(open(args.gen_cache, encoding="utf-8"))
    half = [k for k, v in jc.items() if v[0] == 0.5]
    nonhalf = [k for k, v in jc.items() if v[0] != 0.5]
    print(f"units at exactly 0.5: {len(half)}/{len(jc)} ({len(half)/len(jc):.1%})")
    sample = random.sample(half, min(args.n, len(half)))
    control = random.sample(nonhalf, min(args.n // 3, len(nonhalf)))   # control: non-0.5 units

    convs = json.load(open("results/ulysses_mtrag_100.json", encoding="utf-8"))["conversations"]
    q_of = {(ci, t["n"]): t["query"] for ci, c in enumerate(convs) for t in c["turns"]}

    print("Loading corpus + dense index (context reconstruction) ...")
    bills, _ = DataLoader().load_all()
    dense = DenseRetriever(bills, model_name=BGE_M3); dense.build_index(cache_path=ULYSSES_CACHE)
    client = get_rag(JUDGE_MODEL)._client

    def judge_raw(prompt):
        try:
            resp = client.chat.completions.create(model=JUDGE_MODEL, messages=[{"role": "user", "content": prompt}],
                                                  temperature=0, max_tokens=20)
            txt = resp.choices[0].message.content.strip()
            m = re.search(r"(0?\.\d+|1\.0|0|1)", txt)
            return ("ok", txt, max(0.0, min(1.0, float(m.group(1)))) if m else None)
        except Exception as e:
            return ("error", type(e).__name__ + ": " + str(e)[:120], None)

    rows = []
    for tag, keys in (("half", sample), ("control", control)):
        for k in keys:
            ci, tn, model = k.split("|"); ci, tn = int(ci), int(tn)
            res = dense.retrieve(q_of[(ci, tn)], top_k=5)
            ctx = "\n\n---\n\n".join(f"### {r.bill.name}\n{r.bill.txt_ementa}" for r in res)
            ans = gc.get(k, "")
            status, raw, score = judge_raw(faith_prompt(ans, ctx))
            rows.append({"group": tag, "key": k, "model": model, "round_score": jc[k][0],
                         "status": status, "raw": raw, "score": score})
            time.sleep(0.2)

    out = {"n_half_total": len(half), "n_total": len(jc), "rows": rows}
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    for tag in ("half", "control"):
        rs = [r for r in rows if r["group"] == tag]
        errs = sum(1 for r in rs if r["status"] == "error")
        parsed = [r["score"] for r in rs if r["score"] is not None]
        same = sum(1 for r in rs if r["score"] is not None and abs(r["score"] - r["round_score"]) < 1e-9)
        print(f"\n[{tag}] n={len(rs)} | API errors now: {errs} | unparsable: {sum(1 for r in rs if r['status']=='ok' and r['score'] is None)}")
        print(f"   sequential re-judge == round score: {same}/{len(rs)}")
        print(f"   re-judge score distribution: {dict(sorted(Counter(round(s, 2) for s in parsed).items()))}")
        print(f"   raw reply samples: {[r['raw'][:25] for r in rs[:6]]}")
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

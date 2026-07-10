"""Compare BM25 vs BGE-M3 strategy rankings + paired significance (BM25)."""
import json, random
random.seed(42)
mean = lambda v: sum(v) / len(v)

def load(p):
    return json.load(open(p, encoding="utf-8"))["results"]

dense = load("results/mtrag_eval.json")
bm25 = load("results/mtrag_eval_bm25.json")

def ptest(a, b):
    dif = [x - y for x, y in zip(a, b)]
    obs = abs(mean(dif))
    c = sum(1 for _ in range(10000)
            if abs(mean([x if random.random() < 0.5 else -x for x in dif])) >= obs - 1e-12)
    return mean(dif), (c + 1) / 10001

def conv_means(R, s):
    raw = R[s]["ndcg_raw_by_turn"]
    n = len(raw["T2"])
    return [mean([raw[t][i] for t in ("T2", "T3", "T4")]) for i in range(n)]

for name, R in [("BGE-M3 (dense)", dense), ("BM25 (lexical)", bm25)]:
    print(f"\n=== {name}: mean nDCG@10 (overall over T2-T4) ===")
    order = sorted(R, key=lambda s: -mean(conv_means(R, s)))
    for s in order:
        print(f"  {s:<10} T2-T4 mean = {mean(conv_means(R, s)):.4f}   best={'*' if s==order[0] else ''}")

print("\n=== BM25: full-history (questions) vs last-turn, paired (T2-T4) ===")
d, p = ptest(conv_means(bm25, "questions"), conv_means(bm25, "lastturn"))
print(f"  full-history - last-turn: diff={d:+.4f}  permutation p={p:.4f}")
try:
    from scipy.stats import wilcoxon
    a = conv_means(bm25, "questions"); b = conv_means(bm25, "lastturn")
    dif = [x - y for x, y in zip(a, b) if x != y]
    print(f"  Wilcoxon p={wilcoxon(dif).pvalue:.2e}")
except Exception:
    pass

print("\n=== per-turn nDCG@10, both retrievers, last-turn vs full-history ===")
for tn in ("T1", "T2", "T3", "T4"):
    print(f"  {tn}: dense last={dense['lastturn']['ndcg_by_turn'][tn]:.3f} "
          f"full-hist={dense['questions']['ndcg_by_turn'][tn]:.3f}  ||  "
          f"BM25 last={bm25['lastturn']['ndcg_by_turn'][tn]:.3f} "
          f"full-hist={bm25['questions']['ndcg_by_turn'][tn]:.3f}")

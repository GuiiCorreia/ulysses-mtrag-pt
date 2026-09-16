"""Statistical layer for the ICTAI paper: 95% CIs + paired tests.

Retrieval (results/mtrag_eval.json, ndcg_raw_by_turn): per-strategy/turn bootstrap CIs;
paired Wilcoxon (scipy if available, else sign-flip permutation) of last-turn vs each
history-aware strategy on per-conversation mean nDCG over T2-T4.

Generation (results/mtrag_gen.json, per_unit): per-model faithfulness CIs; paired tests
Qwen3-8B vs Gemma-4-31B (plateau) and T1 vs T4 faithfulness (degradation).

    uv run python analyze_significance_mtrag.py
"""
import json, random

random.seed(42)

def mean(xs): return sum(xs) / len(xs)

def boot_ci(xs, B=10000):
    n = len(xs); ms = []
    for _ in range(B):
        ms.append(mean([xs[random.randrange(n)] for _ in range(n)]))
    ms.sort()
    return ms[int(0.025 * B)], ms[int(0.975 * B)]

def paired_test(a, b, B=10000):
    """Two-sided sign-flip permutation test on paired differences (fallback-free)."""
    d = [x - y for x, y in zip(a, b)]
    obs = abs(mean(d))
    cnt = 0
    for _ in range(B):
        s = mean([x if random.random() < 0.5 else -x for x in d])
        if abs(s) >= obs - 1e-12:
            cnt += 1
    p_perm = (cnt + 1) / (B + 1)
    p_wilc = None
    try:
        from scipy.stats import wilcoxon
        nz = [x for x in d if x != 0]
        if nz:
            p_wilc = wilcoxon(nz).pvalue
    except Exception:
        pass
    return mean(d), p_perm, p_wilc

# ================= RETRIEVAL =================
import sys as _sys
# Optional positional overrides (additive): analyze_significance_mtrag.py [EVAL_JSON] [GEN_JSON]
_EVAL = _sys.argv[1] if len(_sys.argv) > 1 else "results/mtrag_eval.json"
_GEN = _sys.argv[2] if len(_sys.argv) > 2 else "results/mtrag_gen.json"
print(f"[inputs] eval={_EVAL}  gen={_GEN}")
ev = json.load(open(_EVAL, encoding="utf-8"))["results"]
turns = ["T1", "T2", "T3", "T4"]
print("=" * 72, "\nRETRIEVAL — mean nDCG@10 [95% bootstrap CI] per strategy/turn\n" + "=" * 72)
for s in ev:
    row = []
    for t in turns:
        xs = ev[s]["ndcg_raw_by_turn"][t]
        lo, hi = boot_ci(xs)
        row.append(f"{t}={mean(xs):.3f} [{lo:.3f},{hi:.3f}]")
    print(f"  {s:<10} " + "  ".join(row))

print("\nPaired tests: last-turn vs each strategy (per-conv mean over T2-T4)")
def conv_means(s):
    raw = ev[s]["ndcg_raw_by_turn"]
    n = len(raw["T2"])
    return [mean([raw[t][i] for t in ("T2", "T3", "T4")]) for i in range(n)]
base = conv_means("lastturn")
pvals = {}
for s in ev:
    if s == "lastturn": continue
    d, pp, pw = paired_test(base, conv_means(s))
    pvals[s] = pp
    w = f", Wilcoxon p={pw:.4f}" if pw is not None else ""
    print(f"  lastturn - {s:<9}: mean diff={d:+.4f}, permutation p={pp:.4f}{w}")
# Holm correction
hs = sorted(pvals.items(), key=lambda kv: kv[1])
m = len(hs)
print("  Holm-adjusted:", ", ".join(f"{k}: p_adj={min(1.0, (m - i) * p):.4f}" for i, (k, p) in enumerate(hs)))

# degradation significance (lastturn T1 vs T3)
raw = ev["lastturn"]["ndcg_raw_by_turn"]
d, pp, pw = paired_test(raw["T1"], raw["T3"])
print(f"\nDegradation (lastturn): T1 vs T3 mean diff={d:+.4f}, permutation p={pp:.4f}"
      + (f", Wilcoxon p={pw:.2e}" if pw else ""))

# ================= GENERATION =================
try:
    gen = json.load(open(_GEN, encoding="utf-8"))["results"]
    m0 = list(gen)[0]
    assert "per_unit" in gen[m0]
except Exception as e:
    print("\n(GENERATION per_unit indisponível — rode o re-run)", e); raise SystemExit

# "__oracle__" is the 72B scale anchor: faith-only per_unit (no bert/rouge). It is
# reported separately (analyze_scale.py) and skipped in the cross-model loops here.
models = [m for m in gen if m != "__oracle__"]

print("\n" + "=" * 72, "\nGENERATION — faithfulness mean [95% CI] per model\n" + "=" * 72)
faith = {m: [u["faith"] for u in gen[m]["per_unit"]] for m in models}
for m in sorted(models, key=lambda x: -mean(faith[x])):
    lo, hi = boot_ci(faith[m])
    print(f"  {m.split('/')[-1][:24]:<26} {mean(faith[m]):.3f} [{lo:.3f},{hi:.3f}]  (n={len(faith[m])})")
if "__oracle__" in gen and "per_unit" in gen["__oracle__"]:
    of = [u["faith"] for u in gen["__oracle__"]["per_unit"]]
    lo, hi = boot_ci(of)
    print(f"  {'oracle (' + gen['__oracle__']['model'].split('/')[-1][:16] + ')':<26} "
          f"{mean(of):.3f} [{lo:.3f},{hi:.3f}]  (n={len(of)}, faith only)")

def units(m, key):
    return [u[key] for u in gen[m]["per_unit"]]

q3 = [m for m in models if "qwen3-8b" in m.lower()][0]
g31 = [m for m in models if "gemma-4" in m.lower()][0]
d, pp, pw = paired_test(units(q3, "faith"), units(g31, "faith"))
print(f"\nPlateau: Qwen3-8B vs Gemma-4-31B faith: diff={d:+.4f}, permutation p={pp:.4f}"
      + (f", Wilcoxon p={pw:.2e}" if pw else ""))

# per-model T1 vs T4 degradation on faith
print("\nT1 vs T4 faithfulness (paired within model):")
for m in models:
    t1 = [u["faith"] for u in gen[m]["per_unit"] if u["tn"] == 1]
    t4 = [u["faith"] for u in gen[m]["per_unit"] if u["tn"] == 4]
    d, pp, pw = paired_test(t1, t4)
    print(f"  {m.split('/')[-1][:24]:<26} diff(T1-T4)={d:+.4f}, p={pp:.4f}")

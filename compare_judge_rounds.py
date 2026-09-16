"""Round 1 vs round 2 of the faithfulness judge on the SAME generations (Threat iv).

Quantifies judge run-to-run variability behind the paper's "≈0.05" statement:
per-model mean shift, per-unit agreement, ranking stability, and whether the
headline comparisons (Qwen3-8B vs Gemma-4-31B; Qwen 7B/8B/72B ladder; T1->T4
decline) hold in the second round.

    uv run python compare_judge_rounds.py \
        --r1 results/mtrag_gen.json --r2 results/mtrag_gen_round2.json \
        --out results/judge_rounds_comparison.json
"""
import sys, json, random, argparse
from statistics import mean, pstdev

random.seed(42)


def ptest(a, b, n=10000):
    """Paired sign-flip permutation test on a-b (two-sided)."""
    dif = [x - y for x, y in zip(a, b)]
    obs = abs(mean(dif))
    c = sum(1 for _ in range(n)
            if abs(mean([x if random.random() < 0.5 else -x for x in dif])) >= obs - 1e-12)
    return mean(dif), (c + 1) / (n + 1)


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(x, y):
    mx, my = mean(x), mean(y)
    sx, sy = pstdev(x), pstdev(y)
    if sx == 0 or sy == 0:
        return float("nan")
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (len(x) * sx * sy)


def spearman(x, y):
    return pearson(rank(x), rank(y))


def per_unit_map(entry):
    return {(u["ci"], u["tn"]): u["faith"] for u in entry["per_unit"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--r1", default="results/mtrag_gen.json")
    ap.add_argument("--r2", default="results/mtrag_gen_round2.json")
    ap.add_argument("--out", default="results/judge_rounds_comparison.json")
    args = ap.parse_args()

    R1 = json.load(open(args.r1, encoding="utf-8"))["results"]
    R2 = json.load(open(args.r2, encoding="utf-8"))["results"]
    models = [m for m in R1 if m in R2 and "per_unit" in R1[m] and "per_unit" in R2[m]]
    short = lambda m: ("oracle:" + R1[m]["model"].split("/")[-1][:14]) if m == "__oracle__" else m.split("/")[-1][:22]

    out = {"models": {}, "ranking": {}, "headline": {}}
    print("=" * 78)
    print("JUDGE ROUND 1 vs ROUND 2 — same generations, fresh judging (Llama-3.3-70B, temp 0)")
    print("=" * 78)
    print(f"{'model':<24}{'r1':>7}{'r2':>7}{'delta':>8}{'|d| unit':>10}{'same':>7}{'rho':>7}{'p(r1=r2)':>10}")
    means1, means2 = {}, {}
    for m in models:
        u1, u2 = per_unit_map(R1[m]), per_unit_map(R2[m])
        keys = sorted(set(u1) & set(u2))
        a = [u1[k] for k in keys]; b = [u2[k] for k in keys]
        d, p = ptest(b, a)
        mad = mean(abs(x - y) for x, y in zip(a, b))
        same = sum(1 for x, y in zip(a, b) if abs(x - y) < 1e-9) / len(keys)
        rho = spearman(a, b)
        means1[m], means2[m] = mean(a), mean(b)
        by_turn = {}
        for t in (1, 2, 3, 4):
            ka = [u1[k] for k in keys if k[1] == t]; kb = [u2[k] for k in keys if k[1] == t]
            by_turn[f"T{t}"] = {"r1": round(mean(ka), 4), "r2": round(mean(kb), 4), "delta": round(mean(kb) - mean(ka), 4)}
        out["models"][m] = {"n": len(keys), "mean_r1": round(mean(a), 4), "mean_r2": round(mean(b), 4),
                            "delta_mean": round(d, 4), "p_shift": round(p, 4), "mean_abs_unit_diff": round(mad, 4),
                            "frac_identical": round(same, 4), "spearman_units": round(rho, 4), "by_turn": by_turn}
        print(f"{short(m):<24}{mean(a):>7.3f}{mean(b):>7.3f}{d:>+8.3f}{mad:>10.3f}{same:>7.2f}{rho:>7.2f}{p:>10.4f}")

    slms = [m for m in models if m != "__oracle__"]
    r1_rank = sorted(slms, key=lambda m: -means1[m]); r2_rank = sorted(slms, key=lambda m: -means2[m])
    rho_rank = spearman([means1[m] for m in slms], [means2[m] for m in slms])
    max_shift = max(abs(means2[m] - means1[m]) for m in models)
    out["ranking"] = {"round1": [short(m) for m in r1_rank], "round2": [short(m) for m in r2_rank],
                      "identical_order": r1_rank == r2_rank, "spearman_model_means": round(rho_rank, 4),
                      "max_abs_mean_shift": round(max_shift, 4)}
    print("\nRanking r1:", " > ".join(short(m) for m in r1_rank))
    print("Ranking r2:", " > ".join(short(m) for m in r2_rank))
    print(f"identical order: {r1_rank == r2_rank} | Spearman(model means) = {rho_rank:.3f} | "
          f"max |mean shift| = {max_shift:.3f}  (paper states ≈0.05)")

    # ---- headline comparisons under round 2 ----
    def units_of(R, m):
        return per_unit_map(R[m])
    def paired(R, ma, mb):
        ua, ub = units_of(R, ma), units_of(R, mb)
        keys = sorted(set(ua) & set(ub))
        return ptest([ua[k] for k in keys], [ub[k] for k in keys])
    q3 = next((m for m in slms if "qwen3-8b" in m.lower()), None)
    g31 = next((m for m in slms if "gemma-4" in m.lower()), None)
    q7 = next((m for m in slms if "2.5-7b" in m.lower()), None)
    orc = "__oracle__" if "__oracle__" in models else None
    print("\nHeadline comparisons (diff, permutation p) — round 1 | round 2")
    for name, ma, mb in [("Qwen3-8B vs Gemma-4-31B", q3, g31), ("Qwen 8B vs 72B (ladder)", q3, orc),
                         ("Qwen 7B vs 72B (ladder)", q7, orc), ("Qwen 7B vs 8B (ladder)", q7, q3)]:
        if not (ma and mb):
            continue
        d1, p1 = paired(R1, ma, mb); d2, p2 = paired(R2, ma, mb)
        out["headline"][name] = {"r1": {"diff": round(d1, 4), "p": round(p1, 4)}, "r2": {"diff": round(d2, 4), "p": round(p2, 4)}}
        print(f"  {name:<28} r1: {d1:+.3f} (p={p1:.4f}) | r2: {d2:+.3f} (p={p2:.4f})")
    print("\nT1 -> T4 decline per model (diff T1-T4, p) — round 1 | round 2")
    for m in slms:
        u1, u2 = units_of(R1, m), units_of(R2, m)
        t1a = [u1[k] for k in sorted(u1) if k[1] == 1]; t4a = [u1[k] for k in sorted(u1) if k[1] == 4]
        t1b = [u2[k] for k in sorted(u2) if k[1] == 1]; t4b = [u2[k] for k in sorted(u2) if k[1] == 4]
        d1, p1 = ptest(t1a, t4a); d2, p2 = ptest(t1b, t4b)
        out["models"][m]["t1_minus_t4"] = {"r1": {"diff": round(d1, 4), "p": round(p1, 4)}, "r2": {"diff": round(d2, 4), "p": round(p2, 4)}}
        print(f"  {short(m):<24} r1: {d1:+.3f} (p={p1:.4f}) | r2: {d2:+.3f} (p={p2:.4f})")

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

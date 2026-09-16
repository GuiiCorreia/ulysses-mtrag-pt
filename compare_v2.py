"""Side-by-side comparison of judging rounds (original r1, original-protocol r2, v2 r1, v2 r2).

Handles NA (None) faith values in v2 files: per-model means over non-NA; paired
statistics over units non-NA in BOTH compared rounds. Also tabulates, per model,
what the ORIGINAL exact-0.5 units became under v2 (score distribution).

    uv run python compare_v2.py --out results/judge_rounds_v2_comparison.json
"""
import sys, json, random, argparse
from statistics import mean, pstdev
from collections import Counter

random.seed(42)


def ptest(a, b, n=10000):
    dif = [x - y for x, y in zip(a, b)]
    if not dif:
        return 0.0, 1.0
    obs = abs(mean(dif))
    c = sum(1 for _ in range(n) if abs(mean([x if random.random() < 0.5 else -x for x in dif])) >= obs - 1e-12)
    return mean(dif), (c + 1) / (n + 1)


def rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(x, y):
    rx, ry = rank(x), rank(y); mx, my = mean(rx), mean(ry); sx, sy = pstdev(rx), pstdev(ry)
    return float("nan") if sx == 0 or sy == 0 else sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / (len(x) * sx * sy)


def load(p):
    d = json.load(open(p, encoding="utf-8"))["results"]
    return {m: {(u["ci"], u["tn"]): u["faith"] for u in v["per_unit"]} for m, v in d.items() if "per_unit" in v}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig-r1", default="results/mtrag_gen.json")
    ap.add_argument("--orig-r2", default="results/mtrag_gen_round2.json")
    ap.add_argument("--v2-r1", default="results/mtrag_gen_v2_r1.json")
    ap.add_argument("--v2-r2", default="results/mtrag_gen_v2_r2.json")
    ap.add_argument("--out", default="results/judge_rounds_v2_comparison.json")
    args = ap.parse_args()

    R = {"orig_r1": load(args.orig_r1), "orig_r2": load(args.orig_r2), "v2_r1": load(args.v2_r1)}
    try:
        R["v2_r2"] = load(args.v2_r2)
    except FileNotFoundError:
        print(f"(v2_r2 not found: {args.v2_r2}; comparing what exists)")
    models = [m for m in R["v2_r1"]]
    short = lambda m: "oracle" if m == "__oracle__" else m.split("/")[-1][:22]
    out = {"means": {}, "pairs": {}, "ranking": {}, "orig_half_to_v2": {}}

    # ---- means per model and per turn, every round
    rounds = list(R)
    print("=" * 96)
    print("FAITH mean over non-NA units — " + " | ".join(rounds) + "   (n_na in brackets)")
    print("=" * 96)
    for m in models:
        row = []
        for r in rounds:
            vals = [v for v in R[r].get(m, {}).values()]
            nn = [v for v in vals if v is not None]
            row.append((mean(nn) if nn else float("nan"), len(vals) - len(nn)))
        out["means"].setdefault(m, {})["all"] = {r: {"mean": round(x, 4), "n_na": na} for r, (x, na) in zip(rounds, row)}
        print(f"{short(m):<24}" + "".join(f"{x:>9.3f} [{na:>2}]" for x, na in row))
        for t in (1, 2, 3, 4):
            rowt = []
            for r in rounds:
                vals = [v for (ci, tn), v in R[r].get(m, {}).items() if tn == t]
                nn = [v for v in vals if v is not None]
                rowt.append((mean(nn) if nn else float("nan"), len(vals) - len(nn)))
            out["means"][m][f"T{t}"] = {r: {"mean": round(x, 4), "n_na": na} for r, (x, na) in zip(rounds, rowt)}
            print(f"   T{t:<20}" + "".join(f"{x:>9.3f} [{na:>2}]" for x, na in rowt))

    # ---- pairwise round comparisons (abs diff, per-unit agreement, paired p) on common non-NA units
    def compare(ra, rb):
        res = {}
        print(f"\n--- {ra} vs {rb}: per model |Δmean|, mean|Δunit|, %identical, rho(units), p(shift) ---")
        for m in models:
            A, B = R[ra].get(m, {}), R[rb].get(m, {})
            keys = [k for k in A if k in B and A[k] is not None and B[k] is not None]
            a = [A[k] for k in keys]; b = [B[k] for k in keys]
            if not keys:
                continue
            d, p = ptest(b, a)
            mad = mean(abs(x - y) for x, y in zip(a, b)); same = sum(1 for x, y in zip(a, b) if abs(x - y) < 1e-9) / len(keys)
            rho = spearman(a, b)
            res[m] = {"n_common": len(keys), "mean_a": round(mean(a), 4), "mean_b": round(mean(b), 4),
                      "abs_delta_mean": round(abs(d), 4), "mean_abs_unit_diff": round(mad, 4),
                      "frac_identical": round(same, 4), "spearman_units": round(rho, 4), "p_shift": round(p, 4)}
            print(f"  {short(m):<24} n={len(keys):>3} {mean(a):.3f}->{mean(b):.3f} |Δ|={abs(d):.3f} "
                  f"|Δunit|={mad:.3f} same={same:.0%} rho={rho:.2f} p={p:.4f}")
        slms = [m for m in models if m != "__oracle__"]
        ma = [mean([v for v in R[ra][m].values() if v is not None]) for m in slms]
        mb = [mean([v for v in R[rb][m].values() if v is not None]) for m in slms]
        ra_rank = [short(m) for m in sorted(slms, key=lambda m: -ma[slms.index(m)])]
        rb_rank = [short(m) for m in sorted(slms, key=lambda m: -mb[slms.index(m)])]
        rho_rank = spearman(ma, mb)
        print(f"  ranking {ra}: {' > '.join(ra_rank)}\n  ranking {rb}: {' > '.join(rb_rank)}\n"
              f"  identical order: {ra_rank == rb_rank} | Spearman(model means) = {rho_rank:.3f} | "
              f"max |Δmean| = {max(abs(x - y) for x, y in zip(ma, mb)):.4f}")
        res["_ranking"] = {ra: ra_rank, rb: rb_rank, "identical": ra_rank == rb_rank, "spearman": round(rho_rank, 4)}
        return res
    for ra, rb in [("v2_r1", "v2_r2"), ("orig_r1", "v2_r1"), ("orig_r1", "orig_r2")]:
        if ra in R and rb in R:
            out["pairs"][f"{ra}_vs_{rb}"] = compare(ra, rb)

    # ---- what did the ORIGINAL exact-0.5 units become under v2 r1?
    print("\n--- original r1 units at exactly 0.5 -> v2 r1 value (per model) ---")
    for m in models:
        A, B = R["orig_r1"].get(m, {}), R["v2_r1"].get(m, {})
        keys = [k for k, v in A.items() if v == 0.5 and k in B]
        dist = Counter("NA" if B[k] is None else f"{B[k]:.1f}" for k in keys)
        out["orig_half_to_v2"][m] = {"n_half_orig": len(keys), "v2_dist": dict(sorted(dist.items()))}
        print(f"  {short(m):<24} n(0.5 orig)={len(keys):>3} -> " + ", ".join(f"{k}:{v}" for k, v in sorted(dist.items())))

    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nsaved -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

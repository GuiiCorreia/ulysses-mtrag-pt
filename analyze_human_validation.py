"""RQ2 — human validation of the LLM relevance judge (anonymized annotators A1, A2).

Reads the released, anonymized annotation files and reproduces the human-validation
numbers reported in the paper:

  results/human_eval_A1.csv   110 items of the fixed 200-item sample (A1)
  results/human_eval_A2.csv    41 of those items, doubly annotated (A2)

Each row holds the human label (human_score: 0 / 0.5 / 1) and the judge label
(llm_score) for one (turn query, retrieved bill) pair. Labels are binarized with
"partially relevant" counted as relevant (score >= 0.5).

Reported: raw agreement and Cohen's kappa for judge x A1, judge x A2 and A1 x A2;
95% percentile bootstrap intervals of kappa (10,000 resamples of items, seed 42);
the direction of A1's disagreements with the judge; and A1's labels on the items the
judge rates fully relevant. Standard library only.

    uv run python analyze_human_validation.py
"""
import csv
import random

B, SEED = 10_000, 42


def load(path):
    with open(path, encoding="utf-8") as f:
        return {(r["conv_idx"], r["turn"], r["doc_name"]): r for r in csv.DictReader(f)}


def binar(score):
    return int(float(score) >= 0.5)


def agreement_kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    p1, p2 = sum(x) / n, sum(y) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    return po, kappa


def bootstrap_kappa_ci(x, y, b=B, seed=SEED):
    rng = random.Random(seed)
    n, ks = len(x), []
    for _ in range(b):
        idx = [rng.randrange(n) for _ in range(n)]
        k = agreement_kappa([x[i] for i in idx], [y[i] for i in idx])[1]
        if k == k:  # skip degenerate resamples (a single class on both sides)
            ks.append(k)
    ks.sort()
    return ks[int(0.025 * (len(ks) - 1))], ks[int(0.975 * (len(ks) - 1))]


def main():
    a1, a2 = load("results/human_eval_A1.csv"), load("results/human_eval_A2.csv")

    k1 = sorted(a1)
    h1 = [binar(a1[k]["human_score"]) for k in k1]
    j1 = [binar(a1[k]["llm_score"]) for k in k1]
    po, ka = agreement_kappa(h1, j1)
    lo, hi = bootstrap_kappa_ci(h1, j1)
    print(f"judge x A1  n={len(k1)}  raw agreement {po:.3f}  kappa {ka:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")

    dis = [(h, j) for h, j in zip(h1, j1) if h != j]
    strict = sum(1 for h, j in dis if (h, j) == (1, 0))
    print(f"  A1 disagreements: {len(dis)}; human (at least partially) relevant, judge irrelevant: {strict}")
    fully = [k for k in k1 if float(a1[k]["llm_score"]) == 1.0]
    irrelevant = sum(1 for k in fully if float(a1[k]["human_score"]) == 0.0)
    print(f"  items the judge rates fully relevant: {len(fully)}; rated irrelevant by A1: {irrelevant}")

    k2 = sorted(a2)
    h2 = [binar(a2[k]["human_score"]) for k in k2]
    j2 = [binar(a2[k]["llm_score"]) for k in k2]
    po, ka = agreement_kappa(h2, j2)
    print(f"judge x A2  n={len(k2)}  raw agreement {po:.3f}  kappa {ka:.3f}")

    common = sorted(set(a1) & set(a2))
    x = [binar(a1[k]["human_score"]) for k in common]
    y = [binar(a2[k]["human_score"]) for k in common]
    po, ka = agreement_kappa(x, y)
    lo, hi = bootstrap_kappa_ci(x, y)
    print(f"A1 x A2     n={len(common)}  raw agreement {po:.3f}  kappa {ka:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  (human ceiling)")


if __name__ == "__main__":
    main()

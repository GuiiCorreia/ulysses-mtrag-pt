"""
Agreement measures for judge validation (Ulysses-MTRAG).

Why this module exists
----------------------
The T1 human-vs-judge agreement was previously reported by calling
``compute_fleiss_kappa`` with two raters over the three-level scale
{0.0, 0.5, 1.0}, while the raw-agreement figure reported beside it was
computed on the BINARY projection (relevant iff score >= 0.5). The two
numbers therefore answered different questions and were not comparable:
on the BGE-M3 run they read kappa=0.2026 (ordinal) next to
raw_agreement=0.5770 (binary).

This module reports both readings explicitly, on the same items, plus the
diagnostics needed to interpret them:

  * ``kappa_binary_cohen``     - Cohen's kappa on the binary projection.
                                 This is the headline figure, because it is
                                 the projection on which the judge's gate
                                 decision is actually made.
  * ``kappa_ordinal_fleiss3``  - the previous number, named for what it is.
  * ``pabak_binary``           - prevalence-adjusted, bias-adjusted kappa.
                                 Reported because kappa collapses under
                                 extreme prevalence even at high observed
                                 agreement (the "kappa paradox").
  * ``n_human_labeled`` /
    ``n_pooling_default``      - how many pairs carry an EXPLICIT human label
                                 versus how many were set to 0.0 by the
                                 pooling convention (unjudged = irrelevant).
                                 Without this split the headline kappa is
                                 uninterpretable: it mostly measures the
                                 incompleteness of the source qrels, not the
                                 quality of the judge.
  * ``judged_subset``          - the same measures restricted to the pairs a
                                 human actually saw.
  * ``direction``              - disagreements split into judge-more-lenient
                                 and judge-more-strict. These have different
                                 consequences for benchmark validity and must
                                 not be summed into one error rate.

No measure here changes any previously reported value; the ordinal kappa and
the binary raw agreement are reproduced bit-for-bit. What changes is that the
binary kappa is now reported alongside them instead of being left implicit.

Relation to the other kappa code in this tree
---------------------------------------------
  * ``metrics/kappa.py``            - general matrix-based Fleiss for N raters,
                                      used for multi-judge consistency.
  * ``synthetic/llm_judge.py``      - ``compute_fleiss_kappa``, the original
                                      three-level helper. Still used elsewhere;
                                      ``fleiss_kappa_ordinal`` below reproduces
                                      it exactly for the two-rater case.
  * this module                     - the human-vs-judge T1 report specifically,
                                      where the binary/ordinal distinction and
                                      the pooling split both matter.
"""
from __future__ import annotations

ORDINAL_CATEGORIES = (0.0, 0.5, 1.0)


def _snap(score: float, categories=ORDINAL_CATEGORIES) -> int:
    return min(range(len(categories)), key=lambda i: abs(categories[i] - score))


def fleiss_kappa_ordinal(rater_a: dict, rater_b: dict, items: list) -> float:
    """Fleiss' kappa over the three-level scale. Kept for continuity with the
    previously reported figure; identical to the original implementation."""
    n_items, n_raters, n_cats = len(items), 2, len(ORDINAL_CATEGORIES)
    if n_items == 0:
        return float("nan")
    counts = [[0] * n_cats for _ in range(n_items)]
    for i, item in enumerate(items):
        counts[i][_snap(rater_a.get(item, 0.0))] += 1
        counts[i][_snap(rater_b.get(item, 0.0))] += 1
    total = n_items * n_raters
    p_j = [sum(row[j] for row in counts) / total for j in range(n_cats)]
    p_bar = sum(c * (c - 1) for row in counts for c in row) / (n_items * n_raters * (n_raters - 1))
    p_e = sum(p * p for p in p_j)
    return 1.0 if p_e == 1.0 else (p_bar - p_e) / (1 - p_e)


def cohen_kappa_binary(rater_a: dict, rater_b: dict, items: list, threshold: float = 0.5):
    """Cohen's kappa on the binary projection (relevant iff score >= threshold).

    Returns (raw_agreement, kappa, pabak)."""
    n = len(items)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    a = [rater_a.get(i, 0.0) >= threshold for i in items]
    b = [rater_b.get(i, 0.0) >= threshold for i in items]
    p_o = sum(1 for x, y in zip(a, b) if x == y) / n
    p_a, p_b = sum(a) / n, sum(b) / n
    p_e = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = float("nan") if p_e == 1.0 else (p_o - p_e) / (1 - p_e)
    return p_o, kappa, 2 * p_o - 1


def t1_agreement_report(human: dict, judge: dict, items: list,
                        human_labeled: set | None = None,
                        threshold: float = 0.5) -> dict:
    """Full T1 human-vs-judge report.

    human_labeled: the subset of item keys carrying an EXPLICIT human label.
    Items outside it were set to 0.0 by the pooling convention. Pass None if
    the distinction is unavailable, in which case the subset block is omitted.
    """
    r4 = lambda x: None if x != x else round(x, 4)  # NaN-safe round

    p_o, kappa_bin, pabak = cohen_kappa_binary(human, judge, items, threshold)
    report = {
        "n_judgments": len(items),
        "raw_agreement_binary": r4(p_o),
        "kappa_binary_cohen": r4(kappa_bin),
        "pabak_binary": r4(pabak),
        "kappa_ordinal_fleiss3": r4(fleiss_kappa_ordinal(human, judge, items)),
        # Backward-compatible aliases: these two keys held the previously
        # reported values and keep holding exactly the same numbers.
        "kappa": r4(fleiss_kappa_ordinal(human, judge, items)),
        "raw_agreement": r4(p_o),
    }

    lenient = sum(1 for i in items
                  if judge.get(i, 0.0) >= threshold and human.get(i, 0.0) < threshold)
    strict = sum(1 for i in items
                 if judge.get(i, 0.0) < threshold and human.get(i, 0.0) >= threshold)
    report["direction"] = {"judge_more_lenient": lenient, "judge_more_strict": strict}

    if human_labeled is not None:
        sub = [i for i in items if i in human_labeled]
        report["n_human_labeled"] = len(sub)
        report["n_pooling_default"] = len(items) - len(sub)
        s_po, s_kappa, s_pabak = cohen_kappa_binary(human, judge, sub, threshold)
        report["judged_subset"] = {
            "n": len(sub),
            "raw_agreement_binary": r4(s_po),
            "kappa_binary_cohen": r4(s_kappa),
            "pabak_binary": r4(s_pabak),
            "kappa_ordinal_fleiss3": r4(fleiss_kappa_ordinal(human, judge, sub)),
            "direction": {
                "judge_more_lenient": sum(1 for i in sub if judge.get(i, 0.0) >= threshold
                                          and human.get(i, 0.0) < threshold),
                "judge_more_strict": sum(1 for i in sub if judge.get(i, 0.0) < threshold
                                         and human.get(i, 0.0) >= threshold),
            },
        }
    return report

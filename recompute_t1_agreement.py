#!/usr/bin/env python3
"""
Recompute the T1 human-vs-judge agreement block of an mtrag_eval*.json from the
per-pair judgment dump, without re-running the LLM judge.

Why: the original run reported a three-level Fleiss kappa beside a BINARY raw
agreement (see src/mestrado/metrics/agreement.py). Recomputing the block from
the dump reproduces both legacy values bit-for-bit and adds the binary Cohen
kappa, the PABAK, the pooled/labeled split and the direction of disagreement.
No LLM call is made, so no result is re-estimated: the judge scores are read
from the dump exactly as they were written.

Usage:
    python recompute_t1_agreement.py \
        --dump results/mtrag_judgments.csv \
        --conversations results/ulysses_mtrag_100.json \
        --feedback relevance_feedback_dataset.csv \
        --eval-json results/mtrag_eval.json \
        [--in-place]

Without --in-place it prints the new block and writes nothing.
"""
import argparse
import ast
import csv
import importlib.util
import json
import sys
from pathlib import Path

# Load the agreement module by path rather than through the package. The
# package __init__ pulls in scipy/numpy, and this script must stay runnable on a
# bare interpreter: it regenerates a number that is reported in the paper, so it
# should not depend on the experiment environment being installed. agreement.py
# is pure standard library.
_AGREEMENT = Path(__file__).parent / "src" / "mestrado" / "metrics" / "agreement.py"
_spec = importlib.util.spec_from_file_location("_agreement", _AGREEMENT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
t1_agreement_report = _mod.t1_agreement_report

RELEVANCE_WEIGHTS = {"r": 1.0, "pr": 0.5, "i": 0.0}


def load_human_qrels(path):
    """{query_id: {bill_name: weight}} from relevance_feedback_dataset.csv.

    Mirrors mestrado.data.schema.Query.graded_relevance exactly, including the
    convention that extra_results are implicitly fully relevant."""
    csv.field_size_limit(10_000_000)
    qrels = {}
    with open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            grades = {}
            raw = row.get("user_feedback", "[]")
            if raw.startswith("["):
                try:
                    for d in ast.literal_eval(raw):
                        grades[str(d.get("id", ""))] = RELEVANCE_WEIGHTS[d.get("class", "i")]
                except (ValueError, SyntaxError, KeyError):
                    pass
            raw = row.get("extra_results", "[]")
            if raw.startswith("["):
                try:
                    for bill_id in ast.literal_eval(raw):
                        grades.setdefault(bill_id, 1.0)
                except (ValueError, SyntaxError):
                    pass
            qrels[str(row.get("id", ""))] = grades
    return qrels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True, help="per-pair CSV from --dump-pairs")
    ap.add_argument("--conversations", required=True)
    ap.add_argument("--feedback", required=True)
    ap.add_argument("--eval-json", required=True)
    ap.add_argument("--in-place", action="store_true")
    args = ap.parse_args()

    qrels = load_human_qrels(args.feedback)
    convs = json.load(open(args.conversations, encoding="utf-8"))["conversations"]
    seed_of = [str(c["seed_query_id"]) for c in convs]

    csv.field_size_limit(10_000_000)
    human, judge, labeled = {}, {}, set()
    with open(args.dump, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["turn"]) != 1:
                continue
            ci, name = int(row["conv_idx"]), row["doc_name"]
            key = f"{ci}::{name}"
            grades = qrels.get(seed_of[ci], {})
            if name in grades:
                labeled.add(key)
            human[key] = grades.get(name, 0.0)
            judge[key] = float(row["llm_score"])

    report = t1_agreement_report(human, judge, list(human), labeled)
    report["source"] = ("recomputed from the per-pair dump; judge scores read "
                        "verbatim, no LLM call")

    data = json.load(open(args.eval_json, encoding="utf-8"))
    old = data.get("t1_human_vs_llm", {})
    for key in ("kappa", "raw_agreement", "n_judgments"):
        if key in old and old[key] != report.get(key):
            print(f"  MISMATCH on '{key}': stored {old[key]} vs recomputed "
                  f"{report.get(key)} -- the dump does not belong to this run.",
                  file=sys.stderr)
            return 1
    print("Legacy values reproduced exactly; the dump matches this run.\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.in_place:
        data["t1_human_vs_llm"] = report
        json.dump(data, open(args.eval_json, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        print(f"\nUpdated -> {args.eval_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

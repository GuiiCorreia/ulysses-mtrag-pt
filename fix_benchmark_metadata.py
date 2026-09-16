"""Correct residual metadata in the released benchmark file (ulysses_mtrag_100.json).

The generation run recorded judge_model="phi3:medium" — a leftover default from
the early Ollama-based prototype. That field never drove any evaluation: the
relevance judge actually used is set by --judge-model in run_eval_mtrag.py and
recorded in mtrag_eval*.json["config"] (meta-llama/Llama-3.3-70B-Instruct).
This script fixes the field, records the rewrite model used by the 'rewrite'
strategy, and bumps the version for the public release. Idempotent.

    uv run python fix_benchmark_metadata.py [--file results/ulysses_mtrag_100.json]
"""
import sys, json, argparse

FIX = {
    "judge_model": "meta-llama/Llama-3.3-70B-Instruct",
    "rewrite_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "version": "1.0.0",
}
# The prototype wrote license="CC BY 4.0 (same as base corpus)". The release does not declare a
# license for the corpus (seed queries and bills belong to Ulysses-RFCorpus and are re-linked by
# identifier); licensing of the generated turns/labels/judgments is stated in the repository
# README and LICENSE instead, so the field is removed from the benchmark file.
REMOVE = ["license"]
NOTE = ("metadata corrected for release: judge_model previously held the prototype "
        "default 'phi3:medium'; the judge used in all evaluations is the one recorded "
        "in mtrag_eval*.json config. The 'license' field was removed: see the repository "
        "README/LICENSE (generated turns, labels and judgments: CC BY 4.0; code: MIT; the "
        "Ulysses-RFCorpus seed queries and bills are not redistributed).")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="results/ulysses_mtrag_100.json")
    args = ap.parse_args()

    raw = open(args.file, encoding="utf-8").read()
    ascii_escaped = "\\u" in raw[:200000]
    d = json.loads(raw)
    meta = d.setdefault("metadata", {})
    before = {k: meta.get(k) for k in list(FIX) + REMOVE}
    changed = {k: v for k, v in FIX.items() if meta.get(k) != v}
    removed = [k for k in REMOVE if k in meta]
    for k, v in FIX.items():
        meta[k] = v
    for k in REMOVE:
        meta.pop(k, None)
    meta["metadata_note"] = NOTE
    print("before:", before)
    print("after :", {k: meta.get(k) for k in list(FIX) + REMOVE})
    if not changed and not removed and meta.get("metadata_note") == NOTE:
        print("already up to date; nothing written.")
        return 0
    with open(args.file, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=ascii_escaped, indent=2)
    print(f"WROTE {args.file} (changed: {sorted(changed)}, removed: {removed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Materialize results.__oracle__.per_unit in mtrag_gen.json from the judge cache.

The oracle (Qwen2.5-72B) was scored with --score-oracle, which stored only the
aggregate faithfulness and faith_by_turn. The per-unit scores live in the judge
cache under keys "{ci}|{tn}|{oracle_slug}" -> [faith, fanc]. Without per_unit
the within-family Qwen ladder tests (7B->72B p=0.001, 8B->72B p=0.074) cannot be
audited from the artifact, and analyze_significance_mtrag.py breaks on the
__oracle__ entry.

Gates (nothing is written unless ALL pass):
  G1  exactly 400 units (100 conversations x 4 turns), same (ci,tn) set as the
      reference model's per_unit;
  G2  mean faith == stored results.__oracle__.faithfulness   (|diff| < 5e-5);
  G3  by-turn means == stored results.__oracle__.faith_by_turn (each |diff| < 5e-5).

    uv run python rewrite_oracle_per_unit.py            # gates + write
    uv run python rewrite_oracle_per_unit.py --dry-run  # gates only
"""
import sys, json, argparse
from collections import defaultdict

GEN = "results/mtrag_gen.json"
CACHE = "results/cache/judge_mtrag_gen.json"
REF_MODEL = "qwen/qwen3-8b"       # any evaluated model; defines the unit order


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--gen", default=GEN)
    ap.add_argument("--cache", default=CACHE)
    args = ap.parse_args()

    raw = open(args.gen, encoding="utf-8").read()
    ascii_escaped = "\\u" in raw[:200000]           # preserve the file's escaping style
    gen = json.loads(raw)
    cache = json.load(open(args.cache, encoding="utf-8"))

    R = gen["results"]
    orc = R["__oracle__"]
    slug = orc["model"]
    stored_mean = float(orc["faithfulness"])
    stored_bt = {k: float(v) for k, v in orc["faith_by_turn"].items()}

    units = [(u["ci"], u["tn"]) for u in R[REF_MODEL]["per_unit"]]
    per_unit, missing = [], []
    for ci, tn in units:
        k = f"{ci}|{tn}|{slug}"
        if k not in cache:
            missing.append(k); continue
        per_unit.append({"ci": ci, "tn": tn, "faith": float(cache[k][0])})

    # ---- G1
    ok1 = (len(per_unit) == 400 and not missing and len(set(units)) == 400)
    print(f"G1 units: {len(per_unit)}/400 from cache, missing={len(missing)}, "
          f"distinct (ci,tn)={len(set(units))}  [{'PASS' if ok1 else 'FAIL'}]")
    # ---- G2
    mean = sum(u["faith"] for u in per_unit) / max(1, len(per_unit))
    ok2 = abs(mean - stored_mean) < 5e-5
    print(f"G2 mean faith: {mean:.4f} vs stored {stored_mean}  [{'PASS' if ok2 else 'FAIL'}]")
    # ---- G3
    bt = defaultdict(list)
    for u in per_unit:
        bt[f"T{u['tn']}"].append(u["faith"])
    ok3 = True
    for t in sorted(stored_bt):
        rec = round(sum(bt[t]) / max(1, len(bt[t])), 4)
        good = abs(rec - stored_bt[t]) < 5e-5
        ok3 &= good
        print(f"   {t}: {rec} vs stored {stored_bt[t]}  [{'PASS' if good else 'FAIL'}]")
    print(f"G3 by_turn  [{'PASS' if ok3 else 'FAIL'}]")

    if not (ok1 and ok2 and ok3):
        print("ABORT: gates failed; file untouched.")
        return 1
    if args.dry_run:
        print("DRY RUN: all gates passed; nothing written.")
        return 0

    orc["per_unit"] = per_unit
    orc["per_unit_note"] = ("faith only (reference-less); the oracle has no BERTScore/ROUGE "
                            "because it IS the similarity reference. Materialized from "
                            "results/cache/judge_mtrag_gen.json by rewrite_oracle_per_unit.py.")
    with open(args.gen, "w", encoding="utf-8") as f:
        json.dump(gen, f, ensure_ascii=ascii_escaped, indent=2)
    print(f"WROTE {args.gen}: results.__oracle__.per_unit ({len(per_unit)} units)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

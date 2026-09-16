"""Anonymize human-annotation CSVs before they enter any repository.

human_eval_app.py writes one file per annotator, results/human_eval_<name>.csv,
and stores the name again in the 'annotator' column. Both must become generic
IDs (A1, A2, ...) before release. This script:

  1. collects results/human_eval_*.csv (and annotation_deploy/annotations/*.csv),
     skipping human_eval_sample.csv (the unlabeled sample) and already-anonymized
     files (human_eval_A<n>.csv);
  2. assigns aliases deterministically (sorted by file name) unless --map is given
     (e.g. --map "Fulano=A1,Beltrano=A2");
  3. writes results/human_eval_A1.csv, human_eval_A2.csv, ... with the 'annotator'
     column replaced by the alias;
  4. GATE: re-scans every output for every real name (case-insensitive, whole
     name and each token >= 4 chars) and aborts if any survives.

Real names are never printed. An optional private mapping can be saved with
--map-out (keep it OUT of git; *.private.json is gitignored).

    uv run python anonymize_annotations.py
    uv run python anonymize_annotations.py --map "Nome Um=A1,Nome Dois=A2" --map-out results/annotators.private.json
"""
import sys, csv, re, json, argparse, glob
from pathlib import Path

csv.field_size_limit(10**7)
SAMPLE = "human_eval_sample.csv"
ANON_RE = re.compile(r"^human_eval_A\d+\.csv$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="*", default=None,
                    help="explicit CSVs; default: results/human_eval_*.csv + annotation_deploy/annotations/*.csv")
    ap.add_argument("--map", default=None, help='"Real Name=A1,Other Name=A2"')
    ap.add_argument("--map-out", default=None, help="save the private name->alias map here (never commit)")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    files = args.inputs or (glob.glob("results/human_eval_*.csv")
                            + glob.glob("annotation_deploy/annotations/human_eval_*.csv"))
    files = [f for f in files if Path(f).name != SAMPLE and not ANON_RE.match(Path(f).name)]
    files = sorted(set(files), key=lambda p: Path(p).name.lower())
    if not files:
        print("no annotator CSVs found (nothing to do)."); return 0

    # real name = from the 'annotator' column (fallback: file-name suffix)
    real_names = []
    rows_by_file = {}
    for f in files:
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        rows_by_file[f] = rows
        names = sorted({(r.get("annotator") or "").strip() for r in rows} - {""})
        name = names[0] if len(names) == 1 else Path(f).stem.replace("human_eval_", "")
        real_names.append(name)

    if args.map:
        mapping = dict(kv.split("=", 1) for kv in args.map.split(","))
        mapping = {k.strip(): v.strip() for k, v in mapping.items()}
    else:
        mapping = {n: f"A{i+1}" for i, n in enumerate(real_names)}
    missing = [n for n in real_names if n not in mapping]
    if missing:
        print(f"ABORT: {len(missing)} annotator(s) without alias in --map"); return 1

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for f, name in zip(files, real_names):
        alias = mapping[name]
        rows = rows_by_file[f]
        if not rows:
            print(f"  skip empty: {Path(f).name}"); continue
        cols = list(rows[0].keys())
        for r in rows:
            if "annotator" in r:
                r["annotator"] = alias
        out = out_dir / f"human_eval_{alias}.csv"
        with open(out, "w", newline="", encoding="utf-8") as fo:
            w = csv.DictWriter(fo, fieldnames=cols); w.writeheader(); w.writerows(rows)
        written.append((out, len(rows), alias))

    # ---- GATE: no real name survives anywhere in the outputs
    tokens = set()
    for n in real_names:
        tokens.add(n.lower())
        tokens.update(t.lower() for t in re.split(r"[\s_\-\.]+", n) if len(t) >= 4)
    leaks = 0
    for out, _, _ in written:
        blob = open(out, encoding="utf-8").read().lower()
        for t in tokens:
            if t and t in blob:
                leaks += 1
    if leaks:
        for out, _, _ in written:
            out.unlink(missing_ok=True)
        print(f"ABORT: {leaks} name token(s) still present in outputs; outputs deleted. "
              f"(A real name may appear inside a query/ementa text — inspect manually.)")
        return 1

    for out, n, alias in written:
        print(f"  wrote {out}  ({n} rows, annotator={alias})")
    if args.map_out:
        Path(args.map_out).write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  private map saved -> {args.map_out} (do NOT commit)")
    print(f"GATE PASS: {len(written)} file(s), no annotator name in content.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

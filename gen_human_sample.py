"""Build the fixed 200-item human-validation sample, balanced by turn x judge-score.

Stratifies across the 4 turns and the 3 LLM score buckets (irrelevant/partial/
relevant), oversampling the minority buckets so the human-vs-LLM kappa has power
(a purely random sample would be ~69% 'relevant' and yield an uninformative kappa).
Both annotators load this exact file, so inter-annotator kappa is computed on the
same items. The LLM label is NOT shown to annotators during rating.

    uv run python gen_human_sample.py
Output: results/human_eval_sample.csv  (columns identical to mtrag_judgments.csv)
"""
import csv, random
csv.field_size_limit(10**7)
random.seed(42)

SRC = "results/mtrag_judgments.csv"
OUT = "results/human_eval_sample.csv"
PER_TURN = 50           # 4 turns x 50 = 200
TARGET = {0.0: 17, 0.5: 17, 1.0: 16}   # per-turn quota by score bucket

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
by = {}  # (turn, bucket) -> [rows]
for r in rows:
    try:
        s = float(r["llm_score"])
    except ValueError:
        continue
    b = 0.0 if s < 0.25 else (0.5 if s < 0.75 else 1.0)
    by.setdefault((int(r["turn"]), b), []).append(r)

picked, seen = [], set()
turns = sorted({int(r["turn"]) for r in rows})
for t in turns:
    got = 0
    # fill each bucket to its quota
    for b, q in TARGET.items():
        pool = by.get((t, b), [])[:]
        random.shuffle(pool)
        for r in pool[:q]:
            key = (r["conv_idx"], r["turn"], r["doc_name"])
            if key not in seen:
                seen.add(key); picked.append(r); got += 1
    # top up to PER_TURN from any remaining rows of this turn (mostly 'relevant')
    rest = [r for b in (1.0, 0.5, 0.0) for r in by.get((t, b), [])
            if (r["conv_idx"], r["turn"], r["doc_name"]) not in seen]
    random.shuffle(rest)
    for r in rest:
        if got >= PER_TURN:
            break
        seen.add((r["conv_idx"], r["turn"], r["doc_name"])); picked.append(r); got += 1

random.shuffle(picked)
fields = list(rows[0].keys())
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader(); w.writerows(picked)

# report composition
from collections import Counter
comp = Counter((int(r["turn"]),
                0.0 if float(r["llm_score"]) < 0.25 else (0.5 if float(r["llm_score"]) < 0.75 else 1.0))
               for r in picked)
print(f"Escrito {len(picked)} itens -> {OUT}")
print("Composição (turno, bucket -> n):")
for t in turns:
    line = "  T%d: " % t + "  ".join(f"{b}={comp.get((t,b),0)}" for b in (0.0, 0.5, 1.0))
    print(line)

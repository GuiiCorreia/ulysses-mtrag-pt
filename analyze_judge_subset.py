"""Reviewer-demanded analyses (offline, no API):

A) De-confounded judge agreement: restrict T1 human-vs-LLM comparison to pool docs
   the experts EXPLICITLY labeled (r/pr/i) — the Voorhees incompleteness confound
   vanishes on this subset. Reports kappa, raw agreement, PABAK, class rates.
B) Propagation: correlation between per-turn retrieval nDCG (last-turn) and
   per-turn generation metrics (faith/rouge/bert), per model.

    uv run python analyze_judge_subset.py
"""
import json, csv, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
csv.field_size_limit(10**7)

# ---------- A) judge agreement on explicitly-labeled subset ----------
from mestrado.data.loader import DataLoader

loader = DataLoader()
queries = loader.load_queries()
qmap = {q.query_id: q for q in queries}

convs = json.load(open("results/ulysses_mtrag_100.json", encoding="utf-8"))["conversations"]
seed_of = {ci: str(c["seed_query_id"]) for ci, c in enumerate(convs)}

pairs_all, pairs_lab = [], []  # (human_bin, llm_bin)
for row in csv.DictReader(open("results/mtrag_judgments.csv", encoding="utf-8")):
    if int(row["turn"]) != 1:
        continue
    ci = int(row["conv_idx"])
    q = qmap.get(seed_of.get(ci))
    if not q:
        continue
    gr = q.graded_relevance()
    doc = row["doc_name"]
    llm = float(row["llm_score"]) >= 0.5
    hum_all = gr.get(doc, 0.0) >= 0.5
    pairs_all.append((hum_all, llm))
    if doc in gr:  # expert explicitly labeled this doc (r/pr/i)
        pairs_lab.append((gr[doc] >= 0.5, llm))

def agree_stats(pairs, name):
    n = len(pairs)
    if n == 0:
        print(f"{name}: n=0"); return
    po = sum(1 for h, l in pairs if h == l) / n
    ph = sum(1 for h, _ in pairs if h) / n
    pl = sum(1 for _, l in pairs if l) / n
    pe = ph * pl + (1 - ph) * (1 - pl)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    pabak = 2 * po - 1
    print(f"{name}: n={n}  raw={po:.3f}  kappa={kappa:.3f}  PABAK={pabak:.3f}  "
          f"expert+={ph:.3f}  judge+={pl:.3f}")

print("=" * 70, "\nA) JUDGE vs EXPERT (T1 pool)\n" + "=" * 70)
agree_stats(pairs_all, "ALL pool docs (unlabeled counted irrelevant)  ")
agree_stats(pairs_lab, "EXPLICITLY expert-labeled subset (deconfounded)")

# ---------- B) retrieval -> generation propagation correlation ----------
def pearson(x, y):
    n = len(x); mx = sum(x)/n; my = sum(y)/n
    num = sum((a-mx)*(b-my) for a, b in zip(x, y))
    dx = sum((a-mx)**2 for a in x) ** 0.5
    dy = sum((b-my)**2 for b in y) ** 0.5
    return num / (dx * dy) if dx and dy else 0.0

def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j+1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(x), rank(y))

ev = json.load(open("results/mtrag_eval.json", encoding="utf-8"))["results"]
raw = ev["lastturn"]["ndcg_raw_by_turn"]  # T -> [per-conv]
gen = json.load(open("results/mtrag_gen.json", encoding="utf-8"))["results"]

print("\n" + "=" * 70, "\nB) CORRELATION lastturn nDCG@10 x generation metric (400 units)\n" + "=" * 70)
for m in gen:
    if m == "__oracle__":
        continue
    xs, F, R, B = [], [], [], []
    for u in gen[m]["per_unit"]:
        nd = raw[f"T{u['tn']}"][u["ci"]]
        xs.append(nd); F.append(u["faith"]); R.append(u["rouge"]); B.append(u["bert"])
    print(f"  {m.split('/')[-1][:24]:<26} faith r={pearson(xs,F):+.3f}/rho={spearman(xs,F):+.3f}  "
          f"rouge r={pearson(xs,R):+.3f}/rho={spearman(xs,R):+.3f}  bert r={pearson(xs,B):+.3f}")
# pooled across models
xs, F, R = [], [], []
for m in gen:
    if m == "__oracle__":
        continue
    for u in gen[m]["per_unit"]:
        xs.append(raw[f"T{u['tn']}"][u["ci"]]); F.append(u["faith"]); R.append(u["rouge"])
print(f"  {'POOLED (2000 units)':<26} faith r={pearson(xs,F):+.3f}/rho={spearman(xs,F):+.3f}  "
      f"rouge r={pearson(xs,R):+.3f}/rho={spearman(xs,R):+.3f}")

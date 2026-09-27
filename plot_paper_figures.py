"""Generate the ICTAI paper figures (vector PDF, TrueType fonts).
    uv run python plot_paper_figures.py                       # camera-ready Fig. 2 and Fig. 4
    uv run python plot_paper_figures.py results/mtrag_gen.json _v1   # legacy judge protocol (v1)
Outputs into paper/: fig_retrieval_turns.pdf (Fig. 2) and fig_generation_turns{SUFFIX}.pdf (Fig. 4).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / "paper"
OUT.mkdir(exist_ok=True)
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType (IEEE PDF checker rejects Type 3)
                     "font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "lines.linewidth": 1.3, "lines.markersize": 3.5})
turns = [1, 2, 3, 4]
order = ["lastturn", "concat", "questions", "rewrite"]
labels = {"lastturn": "last-turn", "concat": "concat", "questions": "full-history", "rewrite": "rewrite"}
markers = {"lastturn": "o", "concat": "s", "questions": "^", "rewrite": "D"}

# ---- Fig 1: retrieval nDCG@10 per turn, dense | BM25 (two panels) ----
dense = json.load(open("results/mtrag_eval.json", encoding="utf-8"))["results"]
bm25 = json.load(open("results/mtrag_eval_bm25.json", encoding="utf-8"))["results"]

fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.15), sharey=True)
for ax, (title, R) in zip(axes, [("BGE-M3 (dense)", dense), ("BM25 (lexical)", bm25)]):
    for s in order:
        ys = [R[s]["ndcg_by_turn"][f"T{t}"] for t in turns]
        ax.plot(turns, ys, marker=markers[s], label=labels[s])
    ax.set_title(title, fontsize=8)
    ax.set_xlabel("Turn"); ax.set_xticks(turns)
axes[0].set_ylabel("nDCG@10")
axes[0].set_ylim(0.40, 0.95)
axes[1].legend(frameon=False, fontsize=6.5, loc="upper right")
fig.tight_layout(pad=0.3)
fig.savefig(OUT / "fig_retrieval_turns.pdf"); plt.close(fig)
print("wrote", OUT / "fig_retrieval_turns.pdf")

# ---- Fig 2: generation faithfulness per turn, by model ----
# Optional overrides (additive): plot_paper_figures.py [GEN_JSON] [OUTPUT_SUFFIX]
import sys as _sys
_GEN = _sys.argv[1] if len(_sys.argv) > 1 else "results/mtrag_gen_v2_r1.json"   # judge protocol v2, round 1 (the paper)
_SUF = _sys.argv[2] if len(_sys.argv) > 2 else ""
gen = json.load(open(_GEN, encoding="utf-8"))["results"]
name = {"qwen/qwen3-8b": "Qwen3-8B", "qwen/qwen-2.5-7b-instruct": "Qwen2.5-7B",
        "microsoft/phi-4": "Phi-4", "mistralai/ministral-8b-2512": "Ministral-3-8B",
        "google/gemma-4-31B-it": "Gemma-4-31B"}
mk = ["o", "s", "^", "D", "v"]
fig, ax = plt.subplots(figsize=(3.4, 2.5))
i = 0
for m in gen:
    if m == "__oracle__":
        continue
    ys = [gen[m]["by_turn"][f"T{t}"]["faith"] for t in turns]
    ax.plot(turns, ys, marker=mk[i % len(mk)], label=name.get(m, m.split("/")[-1]))
    i += 1
ax.set_xlabel("Turn"); ax.set_ylabel("Faithfulness"); ax.set_xticks(turns)
ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper right")
ax.set_ylim(0.74, 0.99)
fig.tight_layout(pad=0.3)
fig.savefig(OUT / f"fig_generation_turns{_SUF}.pdf"); plt.close(fig)
print("wrote", OUT / f"fig_generation_turns{_SUF}.pdf", f"(from {_GEN})")

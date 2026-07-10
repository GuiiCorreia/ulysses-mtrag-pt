"""Analyze multi-turn generation results (results/mtrag_gen.json) for the paper."""
import sys, json

path = sys.argv[1] if len(sys.argv) > 1 else "results/mtrag_gen.json"
d = json.load(open(path, encoding="utf-8"))
R = d["results"]
models = list(R)
print(f"n_conv={d['n_conversations']} n_turns={d['n_turns']}  models={len(models)}\n")

# Aggregate table
print("AGGREGATE (overall):")
print(f"  {'model':22s}{'Faith':>8}{'FANC':>8}{'BERT':>8}{'ROUGE':>8}")
for m in sorted(models, key=lambda x: R[x]['faithfulness'], reverse=True):
    r = R[m]
    print(f"  {m.split('/')[-1][:20]:22s}{r['faithfulness']:>8.3f}{r['fanc']:>8.3f}"
          f"{r['bertscore_f1']:>8.3f}{r['rouge_l']:>8.3f}")

print("\nPER-TURN (Faith / BERT / ROUGE), with T1->T4 delta:")
for m in models:
    bt = R[m]["by_turn"]; ts = sorted(bt)
    def row(metric):
        vals = [bt[t][metric] for t in ts]
        return " ".join(f"{v:.3f}" for v in vals), vals[-1] - vals[0]
    fa, dfa = row("faith"); be, dbe = row("bert"); ro, dro = row("rouge")
    print(f"  {m.split('/')[-1][:20]:22s}")
    print(f"     Faith {fa}   d(T1->T4)={dfa:+.3f}")
    print(f"     BERT  {be}   d(T1->T4)={dbe:+.3f}")
    print(f"     ROUGE {ro}   d(T1->T4)={dro:+.3f}")

# Mean across models by turn (aggregate trend)
ts = sorted(R[models[0]]["by_turn"])
print(f"\nMEAN across {len(models)} models, by turn ({' '.join(ts)}):")
for met in ["faith", "bert", "rouge"]:
    vals = [sum(R[m]["by_turn"][t][met] for m in models) / len(models) for t in ts]
    print(f"  {met:6s} " + " ".join(f"{v:.3f}" for v in vals)
          + f"   d(T1->T4)={vals[-1]-vals[0]:+.3f}")

"""Quick descriptive stats of the Ulysses-MTRAG dataset (no LLM, instant).
Reports turn-type and answer-type distributions + avg turns, for the paper's
benchmark-statistics table and to quantify the 'self-contained follow-up' claim.

    uv run python analyze_mtrag_dataset.py results/ulysses_mtrag_100.json
"""
import sys, json
from collections import Counter

path = sys.argv[1] if len(sys.argv) > 1 else "results/ulysses_mtrag_100.json"
data = json.load(open(path, encoding="utf-8"))
convs = data["conversations"]

turn_types = Counter()
answer_types = Counter()
turn_types_by_pos = {}
n_turns = []
for c in convs:
    n_turns.append(len(c["turns"]))
    for t in c["turns"]:
        if t["n"] == 1:
            continue  # turn 1 is the seed; types apply to follow-ups
        turn_types[t["type"]] += 1
        answer_types[t.get("answer_type", "?")] += 1
        turn_types_by_pos.setdefault(t["n"], Counter())[t["type"]] += 1

total_follow = sum(turn_types.values())
print(f"conversations: {len(convs)} | avg turns/conv: {sum(n_turns)/len(n_turns):.2f}")
print(f"follow-up turns (T2+): {total_follow}\n")

print("TURN TYPE (T2+):")
for k, v in turn_types.most_common():
    print(f"  {k:<14} {v:4d}  {100*v/total_follow:5.1f}%")
print("\nANSWER TYPE (T2+):")
for k, v in answer_types.most_common():
    print(f"  {k:<14} {v:4d}  {100*v/total_follow:5.1f}%")

# 'non-standalone' share = a proxy for context-dependent (vs self-contained) turns
ns = turn_types.get("nonstandalone", 0)
print(f"\nnon-standalone (context-dependent by design): {ns} ({100*ns/total_follow:.1f}%)")
print(f"=> ~{100*(total_follow-ns)/total_follow:.0f}% of follow-ups are standalone-typed")

"""Within-family (Qwen) faithfulness scale ladder + significance, for the paper."""
import json, random
random.seed(42)
mean = lambda v: sum(v) / len(v)

d = json.load(open("results/mtrag_gen.json", encoding="utf-8"))
R = d["results"]
oracle = R["__oracle__"]["model"]
cache = json.load(open("results/cache/judge_mtrag_gen.json", encoding="utf-8"))


def faith_of(m):
    if m == oracle:
        out = []
        for u in R["qwen/qwen3-8b"]["per_unit"]:
            k = f"{u['ci']}|{u['tn']}|{oracle}"
            if k in cache:
                out.append(cache[k][0])
        return out
    return [u["faith"] for u in R[m]["per_unit"]]


def ptest(a, b):
    dif = [x - y for x, y in zip(a, b)]
    obs = abs(mean(dif))
    c = sum(1 for _ in range(10000)
            if abs(mean([x if random.random() < 0.5 else -x for x in dif])) >= obs - 1e-12)
    return mean(dif), (c + 1) / 10001


q7, q8, q72 = faith_of("qwen/qwen-2.5-7b-instruct"), faith_of("qwen/qwen3-8b"), faith_of(oracle)
print(f"Qwen ladder faith: 7B={mean(q7):.3f}  8B={mean(q8):.3f}  72B={mean(q72):.3f}")
for (na, a), (nb, b) in [(("7B", q7), ("72B", q72)), (("8B", q8), ("72B", q72)),
                          (("7B", q7), ("8B", q8))]:
    dif, p = ptest(b, a)
    print(f"  {nb} vs {na}: diff={dif:+.4f}  p={p:.4f}")

"""Build a LaTeX worked-example block for conv 1, UTF-8 preserved."""
import json, csv
csv.field_size_limit(10**7)
convs = json.load(open("results/ulysses_mtrag_100.json", encoding="utf-8"))["conversations"]
CI = 1
c = convs[CI]

# top doc per turn from judgments
top = {}
with open("results/mtrag_judgments.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        if int(row["conv_idx"]) != CI:
            continue
        tn = int(row["turn"])
        try: sc = float(row["llm_score"])
        except: sc = 0.0
        if tn not in top or sc > top[tn][0]:
            top[tn] = (sc, row["doc_name"], row["ementa"])

def esc(s):
    for a, b in [("\\", ""), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"),
                 ("_", "\\_"), ("$", "\\$"), ("~", " "), ("^", "")]:
        s = s.replace(a, b)
    return " ".join(s.split())

def trim(s, n=95):
    s = esc(s)
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "\\ldots"

label = {1.0: "relevant", 0.5: "partially relevant", 0.0: "irrelevant"}
tylab = {"initial": "seed, human-labeled", "follow_up": "follow-up",
         "clarification": "clarification", "nonstandalone": "non-standalone",
         "comparative": "comparative", "temporal": "temporal"}

lines = [r"\begin{figure}[t]", r"\small\setlength{\parindent}{0pt}"]
for t in c["turns"]:
    tn = t["n"]; ty = t.get("type", "?")
    sc, name, em = top.get(tn, (0.0, "--", ""))
    lab = label.get(round(sc * 2) / 2, f"{sc:.1f}")
    docnum = esc(name.split("(")[0].strip())
    lines.append(rf"\textbf{{T{tn}}} \emph{{({tylab.get(ty, ty)})}}: ``{trim(t['query'])}''\\")
    lines.append(rf"$\rightarrow$ top: \textbf{{{docnum}}} --- \emph{{{lab}}}\\[3pt]")
lines[-1] = lines[-1].replace(r"\\[3pt]", "")
lines += [
    r"\caption{A worked example from Ulysses-MTRAG: a four-turn session seeded by a real",
    r"Ulysses-RFCorpus query (T1, which additionally carries expert relevance labels)",
    r"with LLM-generated follow-ups (T2--T4). For each turn we show the top retrieved",
    r"bill and its LLM-judge relevance label. Queries are reproduced verbatim in the",
    r"original Portuguese, including typos present in the source corpus. The retrieved",
    r"bill \emph{shifts} across turns and degrades to only \emph{partially relevant} by",
    r"T4.}",
    r"\label{fig:example}",
    r"\end{figure}",
]
open("../paper_ictai/example_snippet.tex", "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))

"""
Human validation page for Ulysses-MTRAG (ICTAI 2026).

Validates the LLM relevance judge against human annotators on a sample of
(turn-query, retrieved-doc) pairs. Each annotator rates relevance (i/pr/r);
the app saves incrementally and computes human-vs-LLM Cohen's kappa and
inter-annotator kappa.

Input  : results/mtrag_judgments.csv  (from: run_eval_mtrag.py --dump-pairs ...)
Output : results/human_eval_<annotator>.csv  (one per annotator)

Run (from codigos_mestrado/):
    uv run streamlit run human_eval_app.py
(if streamlit is missing:  uv add streamlit)
"""
import os
import random
from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:
    cohen_kappa_score = None

LABELS = {"Irrelevante (i)": 0.0, "Parcialmente relevante (pr)": 0.5, "Relevante (r)": 1.0}
RESULTS_DIR = Path("results")


def to_class(score: float) -> int:
    return 0 if score < 0.25 else (1 if score < 0.75 else 2)


@st.cache_data
def load_stratified_sample(path: str, n: int, seed: int) -> pd.DataFrame:
    """Sample ~n/turn-group items, stratified by turn, hiding the LLM label during rating."""
    df = pd.read_csv(path)
    rng = random.Random(seed)
    turns = sorted(df["turn"].unique())
    per = max(1, n // len(turns))
    parts = []
    for t in turns:
        sub = df[df["turn"] == t]
        idx = list(sub.index)
        rng.shuffle(idx)
        parts.append(sub.loc[idx[:per]])
    return pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True)


st.set_page_config(page_title="Validação Humana — Ulysses-MTRAG", layout="wide")
st.title("🧑‍⚖️ Validação Humana — Relevância (Ulysses-MTRAG)")

st.sidebar.header("Configuração")
judg_path = st.sidebar.text_input("CSV da amostra", "results/human_eval_sample.csv")
annotator = st.sidebar.text_input("Seu nome (anotador)", "")
fixed = st.sidebar.checkbox("Arquivo já é a amostra final (fixa)", value=True)
n_sample = st.sidebar.number_input("Tamanho (só se não-fixa)", 12, 400, 200, step=4)
seed = st.sidebar.number_input("Seed (igual para todos os anotadores!)", 0, 9999, 42)
mode = st.sidebar.radio("Modo", ["Avaliar", "Analisar concordância"])

if not annotator.strip():
    st.warning("Digite seu nome na barra lateral para começar.")
    st.stop()
if not os.path.exists(judg_path):
    st.error(f"Arquivo não encontrado: {judg_path}\n\nRode antes:\n"
             "uv run python gen_human_sample.py")
    st.stop()

out_path = RESULTS_DIR / f"human_eval_{annotator.strip().replace(' ', '_')}.csv"
if fixed:
    sample = pd.read_csv(judg_path).sample(frac=1, random_state=int(seed)).reset_index(drop=True)
else:
    sample = load_stratified_sample(judg_path, int(n_sample), int(seed))

# Download the current annotator's CSV (for remote collection from the VPS).
if out_path.exists():
    st.sidebar.download_button(
        "⬇️ Baixar minhas anotações (CSV)", out_path.read_bytes(),
        file_name=out_path.name, mime="text/csv")

# -------------------------------------------------------------------- Avaliar
if mode == "Avaliar":
    done = set()
    if out_path.exists():
        d = pd.read_csv(out_path)
        done = {(r.conv_idx, r.turn, r.doc_name) for r in d.itertuples()}

    remaining = [i for i in range(len(sample))
                 if (sample.loc[i, "conv_idx"], sample.loc[i, "turn"],
                     sample.loc[i, "doc_name"]) not in done]

    st.progress((len(sample) - len(remaining)) / len(sample))
    st.caption(f"{len(sample) - len(remaining)} / {len(sample)} avaliados  ·  anotador: **{annotator}**")

    if not remaining:
        st.success("Tudo avaliado! Vá em **Analisar concordância** na barra lateral.")
        st.stop()

    row = sample.loc[remaining[0]]
    st.markdown(f"### Consulta (turno {int(row['turn'])})")
    st.info(row["query"])
    st.markdown(f"### Documento recuperado: `{row['doc_name']}`")
    st.write("**Ementa:**", row["ementa"])
    with st.expander("Trecho do texto"):
        st.write(row["text_excerpt"])

    st.markdown("**Este documento é relevante para a consulta?**")
    choice = st.radio("Relevância", list(LABELS.keys()), index=0,
                      key=f"r_{row['conv_idx']}_{row['turn']}_{row['doc_name']}",
                      label_visibility="collapsed")

    if st.button("💾 Salvar e próximo", type="primary"):
        rec = {"conv_idx": row["conv_idx"], "turn": row["turn"], "doc_name": row["doc_name"],
               "query": row["query"], "human_score": LABELS[choice], "llm_score": row["llm_score"],
               "annotator": annotator}
        pd.DataFrame([rec]).to_csv(out_path, mode="a", header=not out_path.exists(), index=False)
        st.rerun()

# ----------------------------------------------------------- Analisar
else:
    st.header("Concordância")
    if not out_path.exists():
        st.warning("Você ainda não avaliou nada.")
        st.stop()
    d = pd.read_csv(out_path)
    h = [to_class(x) for x in d["human_score"]]
    l = [to_class(x) for x in d["llm_score"]]
    raw = sum(1 for a, b in zip(h, l) if a == b) / len(h) if h else 0.0
    c1, c2, c3 = st.columns(3)
    c1.metric("n itens", len(h))
    c2.metric("Concordância bruta (humano vs LLM)", f"{raw:.3f}")
    if cohen_kappa_score and len(set(h)) > 1 and len(set(l)) > 1:
        c3.metric("Cohen's κ (humano vs LLM)", f"{cohen_kappa_score(h, l):.3f}")
    else:
        c3.metric("Cohen's κ", "n/d (pouca variação)")

    st.divider()
    st.subheader("Concordância inter-anotador")
    others = [p for p in RESULTS_DIR.glob("human_eval_*.csv")
              if p.resolve() != out_path.resolve()]
    if not others:
        st.caption("Nenhum outro anotador encontrado ainda (precisa de um 2º CSV).")
    for p in others:
        o = pd.read_csv(p)
        merged = d.merge(o, on=["conv_idx", "turn", "doc_name"], suffixes=("_me", "_other"))
        if len(merged) == 0:
            st.caption(f"{p.name}: sem itens em comum.")
            continue
        hm = [to_class(x) for x in merged["human_score_me"]]
        ho = [to_class(x) for x in merged["human_score_other"]]
        raw2 = sum(1 for a, b in zip(hm, ho) if a == b) / len(hm)
        line = f"**{p.name}** — n={len(merged)}  ·  concordância bruta={raw2:.3f}"
        if cohen_kappa_score and len(set(hm)) > 1 and len(set(ho)) > 1:
            line += f"  ·  Cohen's κ={cohen_kappa_score(hm, ho):.3f}"
        st.write(line)

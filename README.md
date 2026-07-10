# Ulysses-MTRAG

Reproduction package for **Ulysses-MTRAG**, a synthetic multi-turn
Retrieval-Augmented Generation (RAG) benchmark for Brazilian legislative dialogue
(Portuguese). Anonymous release accompanying the ICTAI 2026 submission.

## Contents
```
src/mestrado/        core package (data, retrieval, generation, metrics, synthetic, reranking)
run_generate_mtrag.py     build the multi-turn benchmark from the seed corpus
run_eval_mtrag.py         multi-turn retrieval (dense BGE-M3 + lexical BM25, 4 strategies)
run_gen_eval_mtrag.py     generation (5 SLMs + 72B oracle), 3 axes
run_chunking_normastcu.py chunking boundary on long regulatory norms
analyze_*.py              statistics behind the paper (significance, judge audit, scale, BM25)
gen_human_sample.py       fixed 200-item human-validation sample (balanced by turn x score)
human_eval_app.py         Streamlit app for human validation of the LLM judge
annotation_deploy/        Docker + compose to host the annotation app
plot_paper_figures.py     the two paper figures
paper/                    LaTeX source (main.tex, refs, IEEEtran.cls) + figures
results/                  exact metric outputs behind the paper tables/figures
```

## Reproduce
```bash
uv sync
cp .env.example .env          # set DEEPINFRA_API_KEY + OPENROUTER_API_KEY
# retrieval (dense; use --retriever bm25 for the lexical run)
uv run python run_eval_mtrag.py --conversations results/ulysses_mtrag_100.json \
    --provider deepinfra --retriever bge-m3 --k 10 --workers 32 --out results/mtrag_eval.json
# generation (5 SLMs + 72B oracle anchor)
uv run python run_gen_eval_mtrag.py --conversations results/ulysses_mtrag_100.json \
    --workers 32 --out results/mtrag_gen.json --score-oracle
# figures + statistics (from the provided results/)
uv run python plot_paper_figures.py
uv run python analyze_bm25.py && uv run python analyze_scale.py && uv run python analyze_judge_subset.py
```

## Human validation of the LLM judge
The judge is audited against expert labels (see `analyze_judge_subset.py`) and, for a
fresh human study, `gen_human_sample.py` builds a fixed 200-item sample and
`human_eval_app.py` (deployable via `annotation_deploy/`) collects two annotators'
labels, computing human-vs-LLM and inter-annotator agreement.

## Data
The underlying bill corpus (Ulysses-RFCorpus) is a public resource; download it
separately and place `bills_dataset.csv` / `relevance_feedback_dataset.csv` at the
repository root. The `results/` directory ships the final metric JSONs so the paper's
tables and figures can be regenerated without re-issuing the (paid) LLM calls.

## License
Code: MIT.

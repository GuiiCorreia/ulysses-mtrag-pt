# Human-annotation app — VPS deployment

Streamlit app for validating the LLM relevance judge against two human annotators,
over the fixed 200-item sample (`results/human_eval_sample.csv`, balanced by turn ×
judge-score). Both annotators rate the **same** items → human-vs-LLM Cohen's κ +
inter-annotator κ.

## Deploy (VPS with Docker)
```bash
cd codigos_mestrado
mkdir -p annotation_deploy/annotations
cp results/human_eval_sample.csv annotation_deploy/annotations/     # seed the sample
docker compose -f annotation_deploy/docker-compose.yml up -d --build
```
App is served on port **8501**. Put it behind your reverse proxy (Nginx/EasyPanel)
with HTTPS and share the URL with the annotator.

## Annotating
Each annotator, in the sidebar: types a **distinct name**, keeps **seed = 42** and
**"amostra fixa"** checked, then rates each item (irrelevant / partial / relevant).
Progress is saved incrementally.

## Collecting the results
Two ways, both produce `human_eval_<name>.csv`:
- **On the server:** files appear in `annotation_deploy/annotations/`.
- **In the browser:** the annotator clicks **"⬇️ Baixar minhas anotações (CSV)"** and
  sends you the file.

Put both annotators' CSVs into `codigos_mestrado/results/`, then compute agreement:
- open the app in **"Analisar concordância"** mode (shows human-vs-LLM κ + raw
  agreement, and inter-annotator κ once the 2nd CSV is present), or
- run the offline analysis to get the numbers for the paper (RQ2).

## Notes
- The image is CPU-only and tiny (no torch/faiss) — the app only needs
  streamlit + pandas + scikit-learn.
- Do **not** commit the collected `human_eval_*.csv` (they are gitignored).

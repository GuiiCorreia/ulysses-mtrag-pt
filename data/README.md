# data/ — Ulysses-RFCorpus material the 100 sessions draw on

Provenance: **Ulysses-RFCorpus** — Vitório, D.; Souza, E.; Martins, L.; da Silva, N. F. F.;
de Carvalho, A. C. P. L. F.; Oliveira, A. L. I.; de Andrade, F. E. "Building a relevance feedback
corpus for legal information retrieval in the real-case scenario of the Brazilian Chamber of
Deputies." *Language Resources and Evaluation* 59:1257–1277, 2025. doi:10.1007/s10579-024-09767-3.
Redistributed here with attribution, as authorised by one of its authors (see `../NOTICE`).
The rows are copied verbatim from the original files: same columns, same values, no renaming.

| file | records | bytes | SHA-256 |
|---|---|---|---|
| `bills_sessions.csv.gz` (gzip -9 of `bills_sessions.csv`, 104,411,049 bytes) | 4,268 bills | 23,768,738 | gz `2bf55921df26a8ba8039428eefd370a42fc3231fa130cbea73667a3fa7e1e951` · csv `abb8f65b04d12af2a8b902b798fa05ce44c88e4f961b71deaa195d1d04fe3218` |
| `relevance_feedback_sessions.csv` | 100 queries | 147,811 | `32916fa7eadccc84c5fcd874b0bd6b0fdd2cf0abfe7a309e640ce5f4a419cc20` |
| `seed_queries.csv` | 100 | 11,730 | `5693fbe9acd4d86e69acf02e85a529517404169a2fae7b4502b4cc32009508e6` |

Decompress if you need the plain CSV: `gunzip -k bills_sessions.csv.gz` (the loader reads the
`.gz` directly; `build_report.json` records how the subset was built).

## What the subset contains

- `bills_sessions.csv.gz`: every bill cited by the shipped LLM judgments — 3,198 distinct in
  `results/mtrag_judgments.csv` ∪ 981 in `results/mtrag_judgments_bm25.csv` = **3,772** — plus every
  bill referenced by the 100 seed queries' expert labels (`user_feedback` ids and `extra_results`),
  which adds 496 bills, for **4,268** rows. All 3,772 judgment-cited bills are present. 44 identifiers
  cited in the experts' `extra_results` (e.g. `LEI 10671/2003`, `MPV 1027/2021`, `PEC 18/2021`,
  `REQ 399/2022`) are not bills of the 105,669-bill collection in the original release either — they
  are consultants' pointers to other document types — so they cannot be shipped; the loader treats
  them as implicitly relevant identifiers with no text, exactly as with the full collection.
- `relevance_feedback_sessions.csv`: the 100 rows of the original `relevance_feedback_dataset.csv`
  whose `id` is a `seed_query_id` of `results/ulysses_mtrag_100.json` (the 100 sessions' turn-1 queries
  and their expert labels).
- `seed_queries.csv`: `session_id`, `seed_query_id`, `query` (turn-1 text, verbatim).

This subset is sufficient to re-link every identifier in `results/` and to recompute every statistic
in the paper from the shipped judgments. To re-run retrieval over the whole collection (re-index
BM25/BGE-M3), use the complete corpus below.

## Complete corpus

The complete collection (105,669 bills, `bills_dataset.csv`, 1,783,223,874 bytes, SHA-256
`f1672effb35a47969e99880327a2ea734e431fc97edf87e643a5d4f20c465d99`) and the full expert-label file
(`relevance_feedback_dataset.csv`, 692 queries, SHA-256
`9d9408dc8239ec3cb9cfa68e846eb828caac45d318446d25de19b81b3d200163`) are mirrored, unmodified, as a
Hugging Face dataset: https://huggingface.co/datasets/GuiiCorreia/ulysses-rfcorpus-mtrag. Place them in this directory (or set `BILLS_DATASET_PATH` /
`FEEDBACK_DATASET_PATH` in `.env`); the loader prefers the full files when present.

## Columns

`bills_sessions.csv.gz` / `bills_dataset.csv` (one row per bill): `code` (int), `sig_tipo` (str, e.g. `PL`),
`name` (str, e.g. `PL 3650/2021` — the join key used everywhere: `doc_name` in `results/mtrag_judgments*.csv`,
bill identifiers in the judgment caches, `user_feedback[*].id`), `txt_ementa` (str, official summary — what the
dense index embeds), `em_tramitacao`, `situacao` (str), `text` (str, full text), `text_preprocessed` (str).

`relevance_feedback_sessions.csv` / `relevance_feedback_dataset.csv` (one row per expert query):
`id` (int — the benchmark's `seed_query_id`), `query` (str), `user_feedback` (JSON list of
`{"id": "<bill name>", "class": "r"|"pr"|"i", "score", "score_normalized"}`; r = relevant 1.0,
pr = partially relevant 0.5, i = irrelevant 0.0), `extra_results` (JSON list of additional bill ids the
expert marked relevant, weight 1.0), `date_created`, `num_doc_feedback` (int), `extra_results_size` (int).

`seed_queries.csv`: `session_id` (str, e.g. `ulysses_mt_0000`), `seed_query_id` (int), `query` (str).

## Re-linking

- Session → seed query: `results/ulysses_mtrag_100.json[conversations][*].seed_query_id`
  = `relevance_feedback_sessions.csv.id` = `seed_queries.csv.seed_query_id`.
- Judgment → bill: `results/mtrag_judgments*.csv.doc_name` = `bills_sessions.csv.name`.
- Expert label → bill: `user_feedback[*].id` / `extra_results[*]` = `bills_sessions.csv.name`.

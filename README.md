# Ulysses-MTRAG

Benchmark, code and exact experiment outputs for:

> **Ulysses-MTRAG: A Synthetic Multi-Turn Benchmark for Retrieval-Augmented
> Generation over Brazilian Legal Text**
> Guilherme Dutra, André Caraíba, Nádia Silva, Paulo Santos, Sávio Teles,
> Anderson Soares, Allan Silva — Instituto de Informática, Universidade Federal de Goiás (INF/UFG)
> **IEEE ICTAI 2026** (accepted, to appear)

Ulysses-MTRAG is a multi-turn legal RAG benchmark for Brazilian Portuguese: 100
four-turn sessions over the 105,669 bills of Ulysses-RFCorpus. Turn 1 of every session
is a real expert query carrying human relevance judgments; turns 2–4 are LLM-generated
follow-ups of prescribed types; relevance of retrieved bills is labeled by an LLM judge
distinct from the generator, and audited against the expert labels.

---

## What is in this repository

```
results/ulysses_mtrag_100.json      THE BENCHMARK (100 sessions x 4 turns, seed ids, turn types)
data/                               Ulysses-RFCorpus material the sessions draw on: bills, expert relevance labels, seed queries (see data/README.md, NOTICE)
src/mestrado/                       core package (data, retrieval, generation, synthetic, metrics)
run_generate_mtrag.py               build the benchmark from the seed corpus
run_eval_mtrag.py                   multi-turn retrieval: 4 context strategies x {BGE-M3, BM25}, pooled LLM judgments
run_gen_eval_mtrag.py               generation: 5 SLMs + 72B oracle, 3 axes (judge protocol v2 by default)
run_judge_v2.py                     re-judge cached generations with protocol v2 (independent rounds)
run_chunking_normastcu.py           NormasTCU length-boundary study (Table I)
rebuild_bm25_t1_judgments.py        offline reconstruction of the BM25 T1 judgment dump (gated)
analyze_*.py, compare_*.py          every statistic in the paper (see map below)
gen_human_sample.py, human_eval_app.py, annotation_deploy/   human validation of the judge
plot_paper_figures.py               Fig. 2 / Fig. 3 (TrueType fonts)
paper/                              LaTeX source, bibliography, figures (main.tex = camera-ready)
results/, results/cache/            metric outputs, judgment caches (raw judge replies for v2), contexts
```

## Table / figure → script → artifact

| Paper element | Script | Artifact(s) |
|---|---|---|
| Table I — NormasTCU length boundary (bm25, dense_summary, dense_fulldoc, dense_chunked) | `run_chunking_normastcu.py` (`--out`, per-condition merge) | `results/chunking_normastcu_full.json` (bm25 0.3279 / dense_summary 0.0000 / dense_fulldoc 0.2889), `results/chunking_normastcu.json` (dense_chunked 0.3299) |
| Table II — benchmark composition (turn / answer types) | `analyze_mtrag_dataset.py` | `results/ulysses_mtrag_100.json` |
| Table III — multi-turn retrieval nDCG@10, 4 strategies × 2 retrievers | `run_eval_mtrag.py`, `analyze_bm25.py`, `analyze_significance_mtrag.py` | `results/mtrag_eval.json` (BGE-M3), `results/mtrag_eval_bm25.json` (BM25), caches `results/cache/judge_mtrag_eval*.json` |
| RQ2 — judge vs expert labels (raw agreement, κ, PABAK, explicit subset) | `analyze_judge_subset.py`, `rebuild_bm25_t1_judgments.py`, `recompute_t1_agreement.py` | `results/mtrag_judgments.csv`, `results/mtrag_judgments_bm25.csv`, agreement blocks inside `mtrag_eval*.json` |
| Table IV — generation (faithfulness v2, FANC, BERTScore, ROUGE-L) | `run_judge_v2.py` (protocol v2), `analyze_significance_mtrag.py`, `analyze_scale.py`, `compare_v2.py` | `results/mtrag_gen_v2_r1.json`, `results/mtrag_gen_v2_r2.json` (+ `.complete.json`), caches `results/cache/judge_faith_v2_r{1,2}.json`, `results/judge_rounds_v2_comparison.json` |
| Fig. 1 — worked example | `gen_example_snippet.py` | `paper/example_snippet.tex` |
| Fig. 2 — retrieval nDCG@10 by turn (dense / BM25) | `plot_paper_figures.py` | `paper/fig_retrieval_turns.pdf` |
| Fig. 3 — generation faithfulness by turn | `plot_paper_figures.py results/mtrag_gen_v2_r1.json _v2r1` | `paper/fig_generation_turns_v2r1.pdf` |

Original-protocol generation outputs are kept for provenance:
`results/mtrag_gen.json` (round 1) and `results/mtrag_gen_round2.json` (round 2).

## Judge protocol v2 (what changed and why)

The faithfulness/FANC judge (Llama-3.3-70B, temperature 0) originally ran with
`max_tokens=20`, and any reply without a parseable number was silently stored as **0.5**.
The judge frequently opens with a sentence ("Avaliando a fidelidade da resposta…") that
consumed the 20-token budget, so 21–27% of the stored faithfulness scores (and ~90% of
FANC scores) were parse fallbacks rather than judgments. Protocol **v2** — the default of
`run_gen_eval_mtrag.py` (`--judge-protocol v2`) and the protocol used by `run_judge_v2.py`
for the numbers in the paper — keeps the same prompts, judge and temperature and adds:
one final instruction line (*Responda apenas com um número entre 0.0 e 1.0.*),
`max_tokens=64`, a first-number parser (decimal comma accepted), **NA instead of any
fallback** when no number is returned, and up to 3 attempts with backoff on API errors.
Two independent v2 rounds over the same cached generations yielded 0 NA out of 4,400
judgments each, a maximum per-model mean difference of 0.003 between rounds and an
identical model ranking. `--judge-protocol v1` reproduces the legacy behaviour.

## Reproduce

```bash
uv sync                          # Python 3.12
cp .env.example .env             # DEEPINFRA_API_KEY (+ OPENROUTER_API_KEY)
# data/ ships the session subset of the corpus (enough to reproduce every statistic in results/);
# to re-index the whole collection, download the complete corpus from the Hugging Face mirror into data/ (see data/README.md)

# Retrieval (Table III): dense, then lexical
uv run python run_eval_mtrag.py --conversations results/ulysses_mtrag_100.json --provider deepinfra \
    --judge-model meta-llama/Llama-3.3-70B-Instruct --rewrite-model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --retriever bge-m3 --k 10 --workers 32 --out results/mtrag_eval.json --dump-pairs results/mtrag_judgments.csv
uv run python run_eval_mtrag.py ... --retriever bm25 --out results/mtrag_eval_bm25.json

# Generation (Table IV): generations are cached; judging uses protocol v2 by default
uv run python run_gen_eval_mtrag.py --conversations results/ulysses_mtrag_100.json --workers 32 \
    --out results/mtrag_gen_v2.json --score-oracle
# or re-judge the shipped generations in independent rounds (what the paper reports)
uv run python run_judge_v2.py --round 1 && uv run python run_judge_v2.py --round 2 --sim-from results/mtrag_gen_v2_r1.json

# Statistics and figures from the shipped results (no API calls)
uv run python analyze_significance_mtrag.py results/mtrag_eval.json results/mtrag_gen_v2_r1.complete.json
uv run python analyze_scale.py results/mtrag_gen_v2_r1.complete.json
uv run python analyze_bm25.py && uv run python analyze_judge_subset.py && uv run python compare_v2.py
uv run python plot_paper_figures.py results/mtrag_gen_v2_r1.json _v2r1
```

## Human validation of the LLM judge

`gen_human_sample.py` builds the fixed 200-item sample (balanced by turn × judge score);
`human_eval_app.py` (deployable with `annotation_deploy/`) collects annotations; annotator
files are anonymized to `A1`, `A2` with `anonymize_annotations.py` before release.

## Data and release form

Release form: the benchmark in full — the generated turns with turn-type and answerability tags, the seed queries, the bills and the experts' relevance labels the 100 sessions draw on, the LLM-judge relevance and faithfulness judgments, the prompts, the code, and the final metric outputs, so every table and figure regenerates without re-issuing the paid LLM calls.

The seed queries, the bills and the experts' relevance labels come from Ulysses-RFCorpus (Vitório et al., *Language Resources and Evaluation* 59:1257–1277, 2025) and are redistributed here with attribution to that release, as authorised by one of its authors; see `NOTICE`. `data/` carries the subset the 100 sessions draw on — the complete 105,669-bill collection is obtained at the source. NormasTCU (Table I only): https://huggingface.co/datasets/ufca-llms/normas-tcu. The complete collection (105,669 bills, 1.7 GB) and the full expert-label file are mirrored, unmodified, as a Hugging Face dataset: https://huggingface.co/datasets/GuiiCorreia/ulysses-rfcorpus-mtrag.

## Citation

```bibtex
@inproceedings{dutra2026ulyssesmtrag,
  title     = {{Ulysses-MTRAG}: A Synthetic Multi-Turn Benchmark for Retrieval-Augmented
               Generation over {B}razilian Legal Text},
  author    = {Dutra, Guilherme and Cara{\'i}ba, Andr{\'e} and Silva, N{\'a}dia and
               Santos, Paulo and Teles, S{\'a}vio and Soares, Anderson and Silva, Allan},
  booktitle = {Proceedings of the 38th IEEE International Conference on Tools with
               Artificial Intelligence (ICTAI)},
  publisher = {IEEE},
  year      = {2026},
  note      = {To appear}
}
```

## Licenses

- **Code**: MIT (see `LICENSE`).
- **Generated turns, turn-type/answerability tags, and our LLM-judge relevance and
  faithfulness judgments** (`results/ulysses_mtrag_100.json`, `results/mtrag_judgments*.csv`,
  `results/cache/judge_*.json`, `results/mtrag_eval*.json`, `results/mtrag_gen*.json`):
  CC BY 4.0.
- **Seed queries, bills and expert relevance labels** (`data/`): from Ulysses-RFCorpus (Vitório et al.), redistributed with attribution as authorised by one of its authors — see `NOTICE`. The terms of the original release apply.

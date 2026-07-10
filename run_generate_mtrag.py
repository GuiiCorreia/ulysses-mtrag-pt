"""
Generate the Ulysses-MTRAG multi-turn dataset (ICTAI 2026).

Provider-agnostic follow-up generation: --provider ollama|deepinfra|openrouter.
Turn 1 = seed query (human ground truth, Conle specialists); Turns 2+ generated.

Run from codigos_mestrado/.

GATE 1 — smoke test (cheap; eyeball the generated follow-ups before scaling):
    uv run python run_generate_mtrag.py --provider openrouter \
        --model "qwen/qwen-2.5-72b-instruct" --n-sessions 5 \
        --out results/mtrag_smoke.json

Full run (after the smoke test looks good):
    uv run python run_generate_mtrag.py --provider openrouter \
        --model "qwen/qwen-2.5-72b-instruct" --n-sessions 100 \
        --out results/ulysses_mtrag_100.json
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from mestrado.data.loader import DataLoader
from mestrado.synthetic.generator import UlyssesMTRAGGenerator


def main():
    ap = argparse.ArgumentParser(description="Generate Ulysses-MTRAG (ICTAI 2026)")
    ap.add_argument("--provider", default="openrouter",
                    choices=["ollama", "deepinfra", "openrouter"])
    ap.add_argument("--model", default="qwen/qwen-2.5-72b-instruct",
                    help="Generator model id (provider-specific slug)")
    ap.add_argument("--n-sessions", type=int, default=5)
    ap.add_argument("--n-turns", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel API workers (e.g., 8-16 for OpenRouter/DeepInfra)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/mtrag_smoke.json")
    args = ap.parse_args()

    print(f"\n{'='*64}\nULYSSES-MTRAG GENERATION\n{'='*64}")
    print(f"provider={args.provider}  model={args.model}  "
          f"n_sessions={args.n_sessions}  n_turns={args.n_turns}")

    print("\nLoading Ulysses-RFCorpus queries ...")
    loader = DataLoader()
    _, queries = loader.load_all()
    print(f"  {len(queries)} queries loaded")

    gen = UlyssesMTRAGGenerator(
        generator_model=args.model,
        provider=args.provider,
        n_turns=args.n_turns,
        random_seed=args.seed,
    )

    emb_cache = Path("results/cache/mtrag_seed_embeddings.pkl")
    emb_cache.parent.mkdir(parents=True, exist_ok=True)

    convs = gen.generate_dataset(
        queries,
        n_sessions=args.n_sessions,
        output_path=Path(args.out),
        embeddings_cache=emb_cache,
        max_workers=args.workers,
    )

    # Quality peek (Gate 1): print the first couple of conversations in full
    print(f"\n{'='*64}\nSMOKE PEEK\n{'='*64}")
    for c in convs[:3]:
        print(f"\nSession {c.session_id} (cluster {c.topic_cluster}):")
        for t in c.turns:
            print(f"  T{t.n} [{t.type}/{t.answer_type}]: {t.query}")

    print(f"\nDone. {len(convs)} conversations -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

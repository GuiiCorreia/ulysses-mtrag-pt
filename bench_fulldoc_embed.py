"""Time estimate for the dense_fulldoc condition (NormasTCU) on the local GPU.

Embeds a SAMPLE of full-text norms (same clone_with_ementa_as_fulltext path and
the same DenseRetriever/BGE-M3 fp16 stack used by run_chunking_normastcu.py) and
extrapolates to the whole corpus. Nothing is cached or written. Tries the default
corpus batch size first (32) and falls back to smaller batches on CUDA OOM, so
the estimate reflects a configuration that actually fits in 4 GB VRAM.

    uv run python bench_fulldoc_embed.py --n 64
"""
import sys, time, argparse, statistics as st
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from mestrado.data.beir_loader import BEIRLoader
from mestrado.retrieval.dense import DenseRetriever
from run_chunking_normastcu import clone_with_ementa_as_fulltext, BGE_M3_MODEL


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="docs to embed for timing")
    ap.add_argument("--max-doc-chars", type=int, default=30_000)
    ap.add_argument("--batches", nargs="+", type=int, default=[32, 8, 2])
    args = ap.parse_args()

    import torch
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    print(f"device={dev}  vram={vram:.1f}GB")

    loader = BEIRLoader("normas-tcu", max_doc_chars=args.max_doc_chars)
    bills, _ = loader.load()
    full = clone_with_ementa_as_fulltext(bills)
    names = list(full)
    lens = [len(full[n].txt_ementa) for n in names]
    N = len(names)
    print(f"corpus={N:,} docs | chars/doc: mean={st.mean(lens):.0f} median={st.median(lens):.0f} "
          f"p90={sorted(lens)[int(.9*N)]} max={max(lens)} | docs>=30k chars (capped): "
          f"{sum(1 for l in lens if l >= args.max_doc_chars)}")

    # representative sample: stratified by length (short/median/long) so the
    # per-doc time reflects the real mix, not only the giants
    order = sorted(range(N), key=lambda i: lens[i])
    idx = [order[int(k * (N - 1) / (args.n - 1))] for k in range(args.n)]
    sample = {names[i]: full[names[i]] for i in idx}
    print(f"sample={len(sample)} docs, chars mean={st.mean(len(b.txt_ementa) for b in sample.values()):.0f}")

    for bs in args.batches:
        try:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            ret = DenseRetriever(sample, model_name=BGE_M3_MODEL, batch_size=bs)
            t0 = time.time()
            ret.build_index(cache_path=None)
            dt = time.time() - t0
            per_doc = dt / len(sample)
            est_h = per_doc * N / 3600
            print(f"\nbatch={bs}: {dt:.1f}s for {len(sample)} docs -> {per_doc:.2f}s/doc "
                  f"-> FULL CORPUS ESTIMATE ~{est_h:.1f} h  (model load included in first run)")
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\nbatch={bs}: CUDA OOM -> trying smaller batch")
                continue
            raise
    return 0


if __name__ == "__main__":
    sys.exit(main())

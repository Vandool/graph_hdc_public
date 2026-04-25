"""
Universal RRWP retrieval sweep across QM9, ZINC, and PubChem variants.

Goal: find ONE parameter set that works well across all datasets.

Constraints:
  - ZINC boundaries: bins ∈ {3..10}, k ∈ {2..16}
  - PubChem boundaries: bins ∈ {6..10}, k ∈ {2..20}
  - QM9: uniform binning (any bins/k)
  - Universal overlap: bins ∈ {6..8}, k ∈ {2..16}

Usage:
    # Default grid
    uv run --extra cuda128 python experiments/rrwp_retrieval/sweep_universal.py --sequential

    # Single dataset
    uv run --extra cuda128 python experiments/rrwp_retrieval/sweep_universal.py --datasets pubchem32 --workers 4

    # Custom grid
    uv run --extra cuda128 python experiments/rrwp_retrieval/sweep_universal.py \\
        --datasets qm9 zinc --dims 512 1024 --depths 2 3 \\
        --k-values "4,8,12" "6,12,18" --bins 8 --beam-sizes 32 \\
        --n-samples 5000 --sequential
"""

import argparse
import itertools
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PYTHON = sys.executable

SCRIPT = str(Path(__file__).parent / "run_rrwp_retrieval.py")
DEFAULT_OUTPUT_DIR = str(
    Path(__file__).parent.parent / "results" / "reconstructions_and_ablations"
)

# ── Default sweep grid ──────────────────────────────────────────────
DEFAULT_DATASETS = ["qm9", "pubchem16", "pubchem32", "pubchem64"]
DEFAULT_DIMS = [1024]
DEFAULT_DEPTHS = [3]
DEFAULT_K_VALUES = [
    "6,10,14",
]
DEFAULT_BINS = [8]
DEFAULT_BEAM_SIZES = [32]
DEFAULT_N_SAMPLES = 5_000
DEFAULT_DECODER = "greedy"
DEFAULT_SEED = 42

DATASET_CHOICES = ["qm9", "zinc", "pubchem16", "pubchem32", "pubchem64"]
DECODER_CHOICES = ["greedy", "pattern_matching"]


def _file_tag(dataset, depth, dim, k_values, num_bins, beam_size, decoder, n_samples):
    """Build the same tag that run_rrwp_retrieval.py uses for filenames."""
    k_str = k_values.replace(",", "_")
    return (
        f"{dataset}_dim{dim}_depth{depth}_k{k_str}_b{num_bins}"
        f"_{decoder}_bs{beam_size}_n{n_samples}"
    )


def is_done(
    dataset,
    depth,
    dim,
    k_values,
    num_bins,
    beam_size,
    *,
    output_dir,
    decoder,
    n_samples,
):
    """Check if this config already has saved results."""
    tag = _file_tag(
        dataset, depth, dim, k_values, num_bins, beam_size, decoder, n_samples
    )
    return (Path(output_dir) / f"{tag}_rrwp.json").exists()


def build_cmd(
    dataset,
    depth,
    dim,
    k_values,
    num_bins,
    beam_size,
    *,
    output_dir,
    decoder,
    n_samples,
    seed,
):
    return [
        PYTHON,
        SCRIPT,
        "--dataset",
        dataset,
        "--hv_dim",
        str(dim),
        "--depth",
        str(depth),
        "--k_values",
        k_values,
        "--num_bins",
        str(num_bins),
        "--n_samples",
        str(n_samples),
        "--decoder",
        decoder,
        "--beam_size",
        str(beam_size),
        "--seed",
        str(seed),
        "--output_dir",
        output_dir,
    ]


def run_one(args):
    """Run a single config. Returns (tag, returncode, elapsed)."""
    (
        idx,
        total,
        dataset,
        depth,
        dim,
        k_vals,
        num_bins,
        beam_size,
        output_dir,
        decoder,
        n_samples,
        seed,
    ) = args
    tag = f"{dataset} depth={depth} dim={dim} k={k_vals} bins={num_bins} bs={beam_size}"
    cmd = build_cmd(
        dataset,
        depth,
        dim,
        k_vals,
        num_bins,
        beam_size,
        output_dir=output_dir,
        decoder=decoder,
        n_samples=n_samples,
        seed=seed,
    )
    t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL(exit {result.returncode})"
    print(f"  [{idx}/{total}] {tag} — {status} ({elapsed:.0f}s)")
    if result.returncode != 0:
        # Print last 10 lines of stderr for debugging
        stderr_lines = (
            result.stderr.decode("utf-8", errors="replace").strip().split("\n")
        )
        for line in stderr_lines[-10:]:
            print(f"    STDERR: {line}")
    return tag, result.returncode, elapsed


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--workers", type=int, default=4, help="Max parallel runs")
    parser.add_argument("--sequential", action="store_true", help="Run one at a time")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=DATASET_CHOICES,
    )
    parser.add_argument("--dims", nargs="+", type=int, default=DEFAULT_DIMS)
    parser.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS)
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=str,
        default=DEFAULT_K_VALUES,
        help='RW step counts; each item is comma-separated, e.g. "4,8,12"',
    )
    parser.add_argument("--bins", nargs="+", type=int, default=DEFAULT_BINS)
    parser.add_argument("--beam-sizes", nargs="+", type=int, default=DEFAULT_BEAM_SIZES)
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    parser.add_argument(
        "--decoder", type=str, default=DEFAULT_DECODER, choices=DECODER_CHOICES
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    cli = parser.parse_args()

    # Order: dataset outermost, then dims, depths, bins, k_values (fastest)
    configs = []
    for dataset in cli.datasets:
        for dim, depth, num_bins, k_vals, beam_size in itertools.product(
            cli.dims,
            cli.depths,
            cli.bins,
            cli.k_values,
            cli.beam_sizes,
        ):
            configs.append((dataset, depth, dim, k_vals, num_bins, beam_size))

    total = len(configs)
    mode = "sequential" if cli.sequential else f"{cli.workers} workers"
    print(f"Universal RRWP Sweep: {total} configurations, {mode}")
    print(f"  datasets:    {cli.datasets}")
    print(f"  dims:        {cli.dims}")
    print(f"  depths:      {cli.depths}")
    print(f"  k_values:    {cli.k_values}")
    print(f"  bins:        {cli.bins}")
    print(f"  beam_sizes:  {cli.beam_sizes}")
    print(f"  samples:     {cli.n_samples}")
    print(f"  decoder:     {cli.decoder}")
    print(f"  seed:        {cli.seed}")
    print(f"  output:      {cli.output_dir}")
    print()

    # Skip completed
    remaining = [
        c
        for c in configs
        if not is_done(
            *c,
            output_dir=cli.output_dir,
            decoder=cli.decoder,
            n_samples=cli.n_samples,
        )
    ]
    skipped = total - len(remaining)
    if skipped:
        print(f"  Skipping {skipped} already-completed configs")
    if not remaining:
        print("  All configs already done!")
        return

    work = [
        (
            i,
            len(remaining),
            *c,
            cli.output_dir,
            cli.decoder,
            cli.n_samples,
            cli.seed,
        )
        for i, c in enumerate(remaining, 1)
    ]

    failed = []
    completed = skipped
    t_start = time.time()

    if cli.sequential:
        for w in work:
            tag, rc, elapsed = run_one(w)
            if rc != 0:
                failed.append((tag, rc))
            else:
                completed += 1
    else:
        with ProcessPoolExecutor(max_workers=cli.workers) as pool:
            futures = {pool.submit(run_one, w): w for w in work}
            for future in as_completed(futures):
                tag, rc, elapsed = future.result()
                if rc != 0:
                    failed.append((tag, rc))
                else:
                    completed += 1

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(
        f"  Sweep complete: {completed}/{total} succeeded ({skipped} skipped), {len(failed)} failed"
    )
    print(f"  Total wall time: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print(f"{'=' * 70}")
    if failed:
        print("\nFailed configurations:")
        for tag, rc in failed:
            print(f"  {tag}  (exit {rc})")


if __name__ == "__main__":
    main()

"""
Universal RRWP retrieval sweep across QM9, ZINC, and PubChem variants.

Goal: find ONE parameter set that works well across all datasets.

Constraints:
  - ZINC boundaries: bins ∈ {3..10}, k ∈ {2..16}
  - PubChem boundaries: bins ∈ {6..10}, k ∈ {2..20}
  - QM9: uniform binning (any bins/k)
  - Universal overlap: bins ∈ {6..8}, k ∈ {2..16}

Usage:
    PATH="$HOME/.local/bin:$PATH" uv run --extra cuda128 python experiments/rrwp_retrieval/sweep_universal.py
    PATH="$HOME/.local/bin:$PATH" uv run --extra cuda128 python experiments/rrwp_retrieval/sweep_universal.py --workers 2
    PATH="$HOME/.local/bin:$PATH" uv run --extra cuda128 python experiments/rrwp_retrieval/sweep_universal.py --datasets zinc qm9
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
OUTPUT_DIR = str(Path(__file__).parent.parent / "results" / "universal_config")

# ── Sweep grid ──────────────────────────────────────────────────────
DATASETS = ["qm9", "pubchem16", "pubchem32", "pubchem64"]

DIMS = [1024]
DEPTHS = [
    # 2,
    3
]  # depth has minimal effect beyond 2
K_VALUES = [
    # "4",
    # "6,12",                    # 2 features, sparse — good capacity ratio
    # "4,8,12",                  # ZINC baseline best (3 features)
    "6,10,14",  # non-standard even spacing
    # "6,12,16",                 # wider spacing (3 features)
    # "4,8,12,16",               # 4 features, standard
    # "6,24"
]
BINS = [
    # 6,
    8
]  # universal overlap between ZINC & PubChem
BEAM_SIZES = [32]  # beam=32 is clearly best from prior sweeps
N_SAMPLES = 10

# Fixed
DECODER = "greedy"
SEED = 42


def _file_tag(dataset, depth, dim, k_values, num_bins, beam_size):
    """Build the same tag that run_rrwp_retrieval.py uses for filenames."""
    k_str = k_values.replace(",", "_")
    return f"{dataset}_dim{dim}_depth{depth}_k{k_str}_b{num_bins}_{DECODER}_bs{beam_size}_n{N_SAMPLES}"


def is_done(dataset, depth, dim, k_values, num_bins, beam_size):
    """Check if this config already has saved results."""
    tag = _file_tag(dataset, depth, dim, k_values, num_bins, beam_size)
    return (Path(OUTPUT_DIR) / f"{tag}_rrwp.json").exists()


def build_cmd(dataset, depth, dim, k_values, num_bins, beam_size):
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
        str(N_SAMPLES),
        "--decoder",
        DECODER,
        "--beam_size",
        str(beam_size),
        "--seed",
        str(SEED),
        "--output_dir",
        OUTPUT_DIR,
    ]


def run_one(args):
    """Run a single config. Returns (tag, returncode, elapsed)."""
    idx, total, dataset, depth, dim, k_vals, num_bins, beam_size = args
    tag = f"{dataset} depth={depth} dim={dim} k={k_vals} bins={num_bins} bs={beam_size}"
    cmd = build_cmd(dataset, depth, dim, k_vals, num_bins, beam_size)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Max parallel runs")
    parser.add_argument("--sequential", action="store_true", help="Run one at a time")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="Which datasets to sweep (default: all three)",
    )
    cli = parser.parse_args()

    datasets = cli.datasets

    # Order: dataset outermost, then dims, depths, bins, k_values (fastest)
    configs = []
    for dataset in datasets:
        for dim, depth, num_bins, k_vals in itertools.product(
            DIMS,
            DEPTHS,
            BINS,
            K_VALUES,
        ):
            configs.append((dataset, depth, dim, k_vals, num_bins, BEAM_SIZES[0]))

    total = len(configs)
    mode = "sequential" if cli.sequential else f"{cli.workers} workers"
    print(f"Universal RRWP Sweep: {total} configurations, {mode}")
    print(f"  datasets:    {datasets}")
    print(f"  dims:        {DIMS}")
    print(f"  depths:      {DEPTHS}")
    print(f"  k_values:    {K_VALUES}")
    print(f"  bins:        {BINS}")
    print(f"  beam_sizes:  {BEAM_SIZES}")
    print(f"  samples:     {N_SAMPLES}")
    print(f"  decoder:     {DECODER}")
    print(f"  output:      {OUTPUT_DIR}")
    print()

    # Skip completed
    remaining = [c for c in configs if not is_done(*c)]
    skipped = total - len(remaining)
    if skipped:
        print(f"  Skipping {skipped} already-completed configs")
    if not remaining:
        print("  All configs already done!")
        return

    work = [(i, len(remaining), *c) for i, c in enumerate(remaining, 1)]

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

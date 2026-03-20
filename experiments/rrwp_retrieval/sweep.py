"""
Full ablation sweep for RRWP retrieval experiment.

Sweeps over:
  - depth:     {2, 3, 4, 5}
  - hv_dim:    {256, 512, 1024, 2048}
  - k_values:  {(4,), (4,8), (4,8,12), (4,8,12,16)}
  - num_bins:  {4, 5}
  - beam_size: {1, 32}

Fixed: dataset=zinc, n_samples=100, decoder=greedy

File naming includes all parameters (including n_samples) so results never overwrite:
  zinc_dim512_depth4_k4_8_b4_greedy_bs32_n200_rrwp_detailed.csv

Usage:
    PATH="$HOME/.local/bin:$PATH" uv run --extra cuda128 python experiments/rrwp_retrieval/sweep.py
    PATH="$HOME/.local/bin:$PATH" uv run --extra cuda128 python experiments/rrwp_retrieval/sweep.py --workers 2
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
OUTPUT_DIR = str(Path(__file__).parent.parent / "results" / "sweep_results")

# ── Sweep grid ──────────────────────────────────────────────────────
# First Sweep
# DEPTHS = [2, 3, 4, 5]
# DIMS = [256, 512, 1024, 2048]
# K_VALUES = ["4", "4,8", "4,8,12", "4,8,12,16"]
# BINS = [4, 5]
# BEAM_SIZES = [1, 32]

# Second Sweep
DIMS = [512, 1024]          # focused range
DEPTHS = [2, 3, 4]               # depth 5 adds nothing
K_VALUES = [
    "4,12",                       # skip k=8 — test sparser RW steps
    "4,8,12",                     # baseline best
    "4,8,16",                     # replace k=12 with k=16
    "2,4,8,12",                   # add very short walks (k=2)
    "4,8,12,20",                  # add longer walks (k=20)
    "6,10,14",                    # non-standard step sizes
]
BINS = [6, 7]                # test much finer quantisation
BEAM_SIZES = [16, 32]            # beam >= 16 is sufficient
N_SAMPLES = 200                  # more samples for statistical power

# Fixed
DATASET = "zinc"
DECODER = "greedy"
SEED = 42


def _file_tag(depth, dim, k_values, num_bins, beam_size):
    """Build the same tag that run_rrwp_retrieval.py uses for filenames."""
    k_str = k_values.replace(",", "_")
    return f"{DATASET}_dim{dim}_depth{depth}_k{k_str}_b{num_bins}_{DECODER}_bs{beam_size}_n{N_SAMPLES}"


def is_done(depth, dim, k_values, num_bins, beam_size):
    """Check if this config already has saved results."""
    tag = _file_tag(depth, dim, k_values, num_bins, beam_size)
    return (Path(OUTPUT_DIR) / f"{tag}_rrwp.json").exists()


def build_cmd(depth, dim, k_values, num_bins, beam_size):
    return [
        PYTHON, SCRIPT,
        "--dataset", DATASET,
        "--hv_dim", str(dim),
        "--depth", str(depth),
        "--k_values", k_values,
        "--num_bins", str(num_bins),
        "--n_samples", str(N_SAMPLES),
        "--decoder", DECODER,
        "--beam_size", str(beam_size),
        "--seed", str(SEED),
        "--output_dir", OUTPUT_DIR,
    ]


def run_one(args):
    """Run a single config. Returns (tag, returncode, elapsed)."""
    idx, total, depth, dim, k_vals, num_bins, beam_size = args
    tag = f"depth={depth} dim={dim} k={k_vals} bins={num_bins} bs={beam_size}"
    cmd = build_cmd(depth, dim, k_vals, num_bins, beam_size)
    t0 = time.time()
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAIL(exit {result.returncode})"
    print(f"  [{idx}/{total}] {tag} — {status} ({elapsed:.0f}s)")
    return tag, result.returncode, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4, help="Max parallel runs (ignored with --sequential)")
    parser.add_argument("--sequential", action="store_true", help="Run configs one at a time instead of in parallel")
    cli = parser.parse_args()

    # Order so K_VALUES varies fastest (innermost) — parallel workers each
    # get a different k_values and trigger different feature scans instead
    # of racing on the same cache key.
    configs = list(itertools.product(DIMS, DEPTHS, BEAM_SIZES, BINS, K_VALUES))
    configs = [(depth, dim, k, b, bs) for dim, depth, bs, b, k in configs]
    total = len(configs)

    mode = "sequential" if cli.sequential else f"{cli.workers} workers"
    print(f"RRWP Retrieval Sweep: {total} configurations, {mode}")
    print(f"  depths:      {DEPTHS}")
    print(f"  dims:        {DIMS}")
    print(f"  k_values:    {K_VALUES}")
    print(f"  bins:        {BINS}")
    print(f"  beam_sizes:  {BEAM_SIZES}")
    print(f"  samples:     {N_SAMPLES}")
    print(f"  decoder:     {DECODER}")
    print()

    # Skip already-completed configs
    remaining = [
        c for c in configs if not is_done(*c)
    ]
    skipped = total - len(remaining)
    if skipped:
        print(f"  Skipping {skipped} already-completed configs")
    if not remaining:
        print("  All configs already done!")
        return

    work = [
        (i, len(remaining), *c)
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
    print(f"\n{'='*70}")
    print(f"  Sweep complete: {completed}/{total} succeeded ({skipped} skipped), {len(failed)} failed")
    print(f"  Total wall time: {total_time:.0f}s ({total_time/60:.1f}min)")
    print(f"{'='*70}")
    if failed:
        print("\nFailed configurations:")
        for tag, rc in failed:
            print(f"  {tag}  (exit {rc})")


if __name__ == "__main__":
    main()

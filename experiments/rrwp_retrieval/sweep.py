"""
Full ablation sweep for RRWP retrieval experiment.

Sweeps over:
  - depth:     {2, 3, 4, 5}
  - hv_dim:    {256, 512, 1024, 2048}
  - k_values:  {(4,), (4,8), (4,8,12), (4,8,12,16)}
  - num_bins:  {4, 5}
  - beam_size: {1, 32}

Fixed: dataset=zinc, n_samples=100, decoder=greedy

File naming includes all parameters so results never overwrite:
  zinc_dim512_depth4_k4_8_b4_greedy_bs32_rrwp_detailed.csv

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
DEPTHS = [2, 3, 4, 5]
DIMS = [256, 512, 1024, 2048]
K_VALUES = ["4", "4,8", "4,8,12", "4,8,12,16"]
BINS = [4, 5]
BEAM_SIZES = [1, 32]

# Fixed
DATASET = "zinc"
N_SAMPLES = 100
DECODER = "greedy"
SEED = 42


def _file_tag(depth, dim, k_values, num_bins, beam_size):
    """Build the same tag that run_rrwp_retrieval.py uses for filenames."""
    k_str = k_values.replace(",", "_")
    return f"{DATASET}_dim{dim}_depth{depth}_k{k_str}_b{num_bins}_{DECODER}_bs{beam_size}"


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
    parser.add_argument("--workers", type=int, default=4, help="Max parallel runs")
    cli = parser.parse_args()

    # Order so (k_values, bins) varies first — avoids multiple workers
    # triggering the same feature scan simultaneously.
    configs = list(itertools.product(K_VALUES, BINS, DEPTHS, DIMS, BEAM_SIZES))
    configs = [(depth, dim, k, b, bs) for k, b, depth, dim, bs in configs]
    total = len(configs)

    print(f"RRWP Retrieval Sweep: {total} configurations, {cli.workers} workers")
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

"""
RRWP-Enriched Retrieval Experiment

Tests whether RRWP (Random Walk Return Probability) features flowing through
the full HDC encoding pipeline (message passing -> edge_terms, graph_embedding)
improve or hurt deterministic graph reconstruction compared to base features only.

Key difference from RRWPHyperNet (split codebook):
  RRWPHyperNet:  base codebook -> message passing -> edge_terms, graph_embedding
                 full codebook (base+RW) -> node_terms only

  This experiment: base+RW codebook -> message passing -> ALL terms enriched

Usage:
    python run_rrwp_retrieval.py --dataset zinc --hv_dim 512 --k_values 6 --num_bins 4 --n_samples 1000
    python run_rrwp_retrieval.py --dataset zinc --n_samples 5   --skip_graph_decode  # smoke test
"""

import argparse
import json
import pickle
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torchhd
from sklearn.model_selection import train_test_split
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric import seed_everything
from tqdm import tqdm

from graph_hdc.datasets.utils import get_split, scan_node_features_with_rw
from graph_hdc.hypernet.configs import (
    DecoderSettings,
    RWConfig,
    create_config_with_rw,
)
from graph_hdc.hypernet.encoder import HyperNet
from graph_hdc.hypernet.types import Feat
from graph_hdc.utils.helpers import DataTransformer, pick_device
from graph_hdc.utils.rw_features import augment_data_with_rw, get_zinc_rw_boundaries

# Re-use baseline config builder from the existing retrieval experiment
from experiments.scripts.run_retrieval_experiment import create_dynamic_config


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def pyg_to_nx_for_ground_truth(data, base_feature_dim: int):
    """Convert PyG data to NX, stripping RRWP columns for compatibility.

    ``DataTransformer.pyg_to_nx`` expects ``data.x`` with 4 or 5 columns.
    When RRWP columns are present we trim them first.
    """
    trimmed = data.clone()
    trimmed.x = data.x[:, :base_feature_dim]
    return DataTransformer.pyg_to_nx(trimmed)


def compute_edge_accuracy(original_edges, decoded_edges):
    """IoU of edge multisets (tuples of arbitrary length)."""
    orig_counter = Counter(original_edges)
    dec_counter = Counter(decoded_edges)
    intersection = sum((orig_counter & dec_counter).values())
    union = sum((orig_counter | dec_counter).values())
    return intersection / union if union > 0 else 0.0


def strip_trailing_dims(tuples_list, keep_dims):
    """Strip each tuple to its first *keep_dims* elements."""
    return [t[:keep_dims] for t in tuples_list]


def _edge_tuples_from_pyg(data):
    """Extract list of (src_feat_tuple, dst_feat_tuple) from PyG data."""
    node_tuples = [tuple(row) for row in data.x.int().tolist()]
    return [
        (node_tuples[u], node_tuples[v])
        for u, v in data.edge_index.t().int().cpu().tolist()
    ]


def graphs_isomorphic_base(g1: nx.Graph, g2: nx.Graph, base_dims: int) -> bool:
    """Isomorphism check comparing only the first *base_dims* features per node.

    Handles both ``feat`` (Feat object) and ``type`` (tuple) node attribute formats.
    """
    if g1.number_of_nodes() != g2.number_of_nodes():
        return False
    if g1.number_of_edges() != g2.number_of_edges():
        return False

    def _base_features(attrs):
        if "feat" in attrs:
            f = attrs["feat"]
            full = (f.atom_type, f.degree_idx, f.formal_charge_idx, f.explicit_hs)
            if f.is_in_ring is not None:
                full = full + (int(f.is_in_ring),)
            return full[:base_dims]
        if "type" in attrs:
            return attrs["type"][:base_dims]
        return None

    def node_match(n1, n2):
        t1 = _base_features(n1)
        t2 = _base_features(n2)
        return t1 == t2 and t1 is not None

    try:
        return nx.is_isomorphic(g1, g2, node_match=node_match)
    except Exception:
        return False


def plot_comparison(
    baseline_df: pd.DataFrame | None,
    rrwp_df: pd.DataFrame,
    output_dir: Path,
    metric: str = "edge_accuracy",
    ylabel: str = "Edge Accuracy",
    tag: str = "",
):
    """Side-by-side bar chart of *metric* by molecule size for baseline vs RRWP."""
    fig, ax = plt.subplots(figsize=(14, 6))

    rrwp_grouped = rrwp_df.groupby("num_nodes")[metric].mean()

    if baseline_df is not None:
        base_grouped = baseline_df.groupby("num_nodes")[metric].mean()
        all_sizes = sorted(set(base_grouped.index) | set(rrwp_grouped.index))
        x = np.arange(len(all_sizes))
        width = 0.35
        ax.bar(
            x - width / 2,
            [base_grouped.get(s, 0) for s in all_sizes],
            width,
            label="Baseline",
            color="steelblue",
            alpha=0.8,
        )
        ax.bar(
            x + width / 2,
            [rrwp_grouped.get(s, 0) for s in all_sizes],
            width,
            label="RRWP",
            color="coral",
            alpha=0.8,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(all_sizes)
    else:
        all_sizes = sorted(rrwp_grouped.index)
        ax.bar(all_sizes, [rrwp_grouped.get(s, 0) for s in all_sizes],
               color="coral", alpha=0.8, label="RRWP")

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Molecule Size")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    plt.tight_layout()
    prefix = f"{tag}_" if tag else ""
    fig.savefig(output_dir / f"{prefix}comparison_{metric}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------

def _cache_path(output_dir: Path, dataset: str, k_values: tuple, num_bins: int) -> Path:
    cache_dir = output_dir / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    k_str = "_".join(str(k) for k in k_values)
    return cache_dir / f"{dataset}_k{k_str}_b{num_bins}.pkl"


def load_or_scan_features(
    output_dir: Path,
    dataset_name: str,
    rw_config: RWConfig,
) -> set[tuple]:
    """Load cached observed features or scan the dataset."""
    cache = _cache_path(output_dir, dataset_name, rw_config.k_values, rw_config.num_bins)
    if cache.is_file():
        print(f"Loading cached features from {cache}")
        with open(cache, "rb") as f:
            return pickle.load(f)

    print("Scanning dataset for observed RW-augmented features (this may take a while)...")
    observed = scan_node_features_with_rw(dataset_name, rw_config)
    with open(cache, "wb") as f:
        pickle.dump(observed, f)
    print(f"Cached {len(observed)} observed feature tuples to {cache}")
    return observed


# ---------------------------------------------------------------------------
# Single-condition runner
# ---------------------------------------------------------------------------

def run_condition(
    *,
    condition_name: str,
    hypernet: HyperNet,
    samples: list,
    base_feature_dim: int,
    dataset_name: str,
    decoder: str,
    beam_size: int,
    skip_graph_decode: bool,
    device: torch.device,
) -> tuple[dict, pd.DataFrame]:
    """Run encoding + decoding for one condition and return (summary, detail_df)."""
    print(f"\n{'=' * 70}")
    print(f"  Condition: {condition_name}")
    print(f"{'=' * 70}")

    config = hypernet.config if hasattr(hypernet, "config") else None

    # --- Phase 1: Batch encoding ---
    batch_size = 256
    loader = DataLoader(samples, batch_size=batch_size, shuffle=False)
    encoded = []

    print(f"Encoding {len(samples)} samples...")
    for batch in tqdm(loader, desc="Encoding"):
        batch = batch.to(device)
        t0 = time.time()
        with torch.no_grad():
            out = hypernet.forward(batch)
        dt = time.time() - t0
        batch_list = batch.to_data_list()
        for i, d in enumerate(batch_list):
            encoded.append({
                "pyg_data": d,
                "edge_term": out["edge_terms"][i],
                "graph_term": out["graph_embedding"][i],
                "encoding_time": dt / len(batch_list),
                "num_nodes": d.num_nodes,
            })

    # --- Phase 2 & 3: Decoding ---
    feature_dim = samples[0].x.size(1)  # total feature dim (may include RRWP)

    edge_acc_full = []
    edge_acc_base = []
    graph_accs = []
    cosine_sims = []
    correction_levels = []
    encoding_times = []
    edge_dec_times = []
    graph_dec_times = []
    num_nodes_list = []

    for rec in tqdm(encoded, desc="Decoding"):
        pyg_data = rec["pyg_data"]
        edge_term = rec["edge_term"]
        graph_term = rec["graph_term"]
        encoding_times.append(rec["encoding_time"])
        num_nodes_list.append(rec["num_nodes"])

        # Ground truth NX graph (base features only)
        nx_gt = pyg_to_nx_for_ground_truth(pyg_data, base_feature_dim)

        # --- Edge decoding ---
        t0 = time.time()
        with torch.no_grad():
            decoded_edges = hypernet.decode_order_one_no_node_terms(edge_term.clone())
        edge_dec_time = time.time() - t0
        edge_dec_times.append(edge_dec_time)

        # Original edges (full features)
        original_edges = _edge_tuples_from_pyg(pyg_data)

        # Edge accuracy: full features
        ea_full = compute_edge_accuracy(original_edges, decoded_edges)
        edge_acc_full.append(ea_full)

        # Edge accuracy: base features only
        orig_base = [(s[:base_feature_dim], d[:base_feature_dim]) for s, d in original_edges]
        dec_base = [(s[:base_feature_dim], d[:base_feature_dim]) for s, d in decoded_edges]
        ea_base = compute_edge_accuracy(orig_base, dec_base)
        edge_acc_base.append(ea_base)

        # --- Graph decoding ---
        if skip_graph_decode:
            graph_accs.append(float("nan"))
            cosine_sims.append(float("nan"))
            correction_levels.append("SKIPPED")
            graph_dec_times.append(0.0)
            continue

        t0 = time.time()
        with torch.no_grad():
            dec_settings = DecoderSettings.get_default_for(
                base_dataset=hypernet.base_dataset,
            )
            if decoder == "greedy":
                dec_settings.fallback_decoder_settings.beam_size = beam_size
                result = hypernet.decode_graph_greedy(
                    edge_term=edge_term,
                    graph_term=graph_term,
                    decoder_settings=dec_settings.fallback_decoder_settings,
                )
            else:
                result = hypernet.decode_graph(
                    edge_term=edge_term,
                    graph_term=graph_term,
                    decoder_settings=dec_settings,
                )
        graph_dec_time = time.time() - t0
        graph_dec_times.append(graph_dec_time)

        correction_levels.append(result.correction_level.name)

        if len(result.nx_graphs) > 0:
            decoded_g = result.nx_graphs[0]
            match = graphs_isomorphic_base(nx_gt, decoded_g, base_feature_dim)
            graph_accs.append(1.0 if match else 0.0)

            # Cosine similarity via re-encoding
            pyg_dec = DataTransformer.nx_to_pyg_with_type_attr(decoded_g)
            batch_dec = Batch.from_data_list([pyg_dec]).to(device)
            with torch.no_grad():
                re_out = hypernet.forward(batch_dec)
            cos = torchhd.cos(graph_term, re_out["graph_embedding"][0]).item()
            cosine_sims.append(cos)
        else:
            graph_accs.append(0.0)
            cosine_sims.append(0.0)

    # --- Summary ---
    corr_counter = Counter(correction_levels)
    n = len(encoded)
    corr_pcts = {k: v / n * 100 for k, v in corr_counter.items()}

    n_graph_hits = sum(1 for g in graph_accs if g == 1.0) if not skip_graph_decode else 0
    n_perfect_edge = sum(1 for e in edge_acc_full if e == 1.0)

    summary = {
        "condition": condition_name,
        "n_samples": n,
        "edge_accuracy_full": float(np.mean(edge_acc_full)),
        "edge_accuracy_base": float(np.mean(edge_acc_base)),
        "perfect_edge_decode": n_perfect_edge,
        "perfect_edge_decode_pct": n_perfect_edge / n * 100,
    }

    if not skip_graph_decode:
        summary.update({
            "graph_accuracy": n_graph_hits / n,
            "graph_hits": n_graph_hits,
            "graph_hits_pct": n_graph_hits / n * 100,
            "cosine_similarity": float(np.mean(cosine_sims)),
            "correction_level_ZERO_pct": corr_pcts.get("ZERO", 0.0),
            "correction_level_ONE_pct": corr_pcts.get("ONE", 0.0),
            "correction_level_TWO_pct": corr_pcts.get("TWO", 0.0),
            "correction_level_THREE_pct": corr_pcts.get("THREE", 0.0),
            "correction_level_FAIL_pct": corr_pcts.get("FAIL", 0.0),
        })

    summary.update({
        "encoding_time_total": float(np.sum(encoding_times)),
        "edge_decoding_time_total": float(np.sum(edge_dec_times)),
        "graph_decoding_time_total": float(np.sum(graph_dec_times)),
    })

    detail_df = pd.DataFrame({
        "condition": condition_name,
        "num_nodes": num_nodes_list,
        "edge_accuracy_full": edge_acc_full,
        "edge_accuracy_base": edge_acc_base,
        "graph_accuracy": graph_accs,
        "cosine_similarity": cosine_sims,
        "correction_level": correction_levels,
        "encoding_time": encoding_times,
        "edge_decoding_time": edge_dec_times,
        "graph_decoding_time": graph_dec_times,
    })

    # Print
    print(f"\n  Edge accuracy (full):    {summary['edge_accuracy_full']:.4f}")
    print(f"  Edge accuracy (base):    {summary['edge_accuracy_base']:.4f}")
    print(f"  Perfect edge decode:     {n_perfect_edge}/{n} ({summary['perfect_edge_decode_pct']:.1f}%)")
    if not skip_graph_decode:
        print(f"  Graph accuracy:          {n_graph_hits}/{n} ({summary['graph_hits_pct']:.1f}%)")
        print(f"  Cosine similarity:       {summary['cosine_similarity']:.4f}")
    print(f"  Encoding time (total):   {summary['encoding_time_total']:.2f} s")
    print(f"  Edge decode time (total):{summary['edge_decoding_time_total']:.2f} s")
    print(f"  Graph decode time (total):{summary['graph_decoding_time_total']:.2f} s")

    return summary, detail_df


def _save_condition(output_dir, tag, experiment_params, condition_name, summary, detail_df):
    """Save one condition's results immediately (JSON, CSV, summary.csv row)."""
    detail_df.to_csv(output_dir / f"{tag}_{condition_name}_detailed.csv", index=False)

    with open(output_dir / f"{tag}_{condition_name}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Append to summary CSV
    row = {**experiment_params, **summary}
    summary_csv = output_dir / "summary.csv"
    df_new = pd.DataFrame([row])
    if summary_csv.exists():
        df_old = pd.read_csv(summary_csv)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(summary_csv, index=False)
    print(f"\n  Saved {condition_name} results to {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RRWP-enriched retrieval experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", type=str, default="zinc", choices=["qm9", "zinc"])
    parser.add_argument("--hv_dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=None, help="Message passing depth (default: dataset-specific)")
    parser.add_argument("--k_values", type=str, default="6", help="Comma-separated RW step counts")
    parser.add_argument("--num_bins", type=int, default=4, help="Quantile bins per RW feature")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--decoder", type=str, default="greedy", choices=["pattern_matching", "greedy"])
    parser.add_argument("--beam_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip_baseline", action="store_true", help="Skip baseline (base features only)")
    parser.add_argument("--skip_graph_decode", action="store_true", help="Skip full graph decode (faster)")

    args = parser.parse_args()
    seed_everything(args.seed)

    k_values = tuple(int(k) for k in args.k_values.split(","))
    depth = args.depth
    if depth is None:
        depth = 4 if args.dataset == "zinc" else 3

    output_dir = Path(__file__).parent.parent / "results" / "rrwp_retrieval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base feature dimension per dataset
    base_feature_dim = 5 if args.dataset == "zinc" else 4

    # Print experiment configuration
    print()
    print("=" * 60)
    print("  RRWP Retrieval Experiment")
    print("=" * 60)
    print(f"  Dataset:          {args.dataset}")
    print(f"  HV dimension:     {args.hv_dim}")
    print(f"  MP depth:         {depth}")
    print(f"  RW k-values:      {k_values}")
    print(f"  RW num_bins:      {args.num_bins}")
    print(f"  Base bins:        {[9,6,3,4,2] if args.dataset == 'zinc' else [4,5,3,5]}")
    print(f"  Extended bins:    {[9,6,3,4,2] + [args.num_bins]*len(k_values) if args.dataset == 'zinc' else [4,5,3,5] + [args.num_bins]*len(k_values)}")
    print(f"  Samples:          {args.n_samples}")
    print(f"  Decoder:          {args.decoder}")
    print(f"  Beam size:        {args.beam_size}")
    print(f"  Seed:             {args.seed}")
    print(f"  Skip baseline:    {args.skip_baseline}")
    print(f"  Skip graph decode:{args.skip_graph_decode}")
    print(f"  Output dir:       {output_dir}")
    print("=" * 60)
    print()

    # -------------------------------------------------------------------
    # 1. Load dataset & stratified sampling
    # -------------------------------------------------------------------
    print("Loading dataset...")
    dataset = get_split(dataset=args.dataset, split="train")
    dataset_size = len(dataset)

    if args.n_samples >= dataset_size:
        sample_indices = list(range(dataset_size))
    else:
        print("Computing molecular sizes for stratified sampling...")
        sizes = np.array([dataset[i].num_nodes for i in tqdm(range(dataset_size), desc="Sizes")])
        bin_labels = pd.qcut(sizes, q=4, labels=False, duplicates="drop")
        n_bins = len(np.unique(bin_labels))
        all_idx = np.arange(dataset_size)
        if args.n_samples >= n_bins:
            sample_indices, _ = train_test_split(all_idx, train_size=args.n_samples, stratify=bin_labels, random_state=args.seed)
        else:
            rng = np.random.RandomState(args.seed)
            sample_indices = rng.choice(all_idx, size=args.n_samples, replace=False)
        sample_indices = sample_indices.tolist()

    print(f"Sampled {len(sample_indices)} molecules from {dataset_size}")

    # Gather base PyG data objects for sampled molecules
    base_samples = [dataset[i].clone() for i in tqdm(sample_indices, desc="Collecting samples")]

    device = pick_device()
    print(f"Device: {device}")

    # -------------------------------------------------------------------
    # 2. Build RW config & scan features
    # -------------------------------------------------------------------
    if args.dataset == "zinc":
        bin_boundaries = get_zinc_rw_boundaries(args.num_bins)
    else:
        bin_boundaries = None  # uniform binning for QM9

    rw_config = RWConfig(
        enabled=True,
        k_values=k_values,
        num_bins=args.num_bins,
        bin_boundaries=bin_boundaries,
    )

    observed = load_or_scan_features(output_dir, args.dataset, rw_config)
    print(f"Observed node feature tuples: {len(observed)}")

    # -------------------------------------------------------------------
    # 3. Create RRWP config & HyperNet (standard HyperNet, extended bins)
    # -------------------------------------------------------------------
    rrwp_config = create_config_with_rw(
        base_dataset=args.dataset,
        hv_dim=args.hv_dim,
        rw_config=rw_config,
        prune_codebook=True,
        hypernet_depth=depth,
    )

    print(f"\nRRWP config bins: {rrwp_config.node_feature_configs}")
    rrwp_hypernet = HyperNet(
        config=rrwp_config,
        depth=depth,
        observed_node_features=observed,
    ).eval().to(device)

    # Codebook stats
    print(f"\n--- Codebook Stats ---")
    print(f"  nodes_codebook: {rrwp_hypernet.nodes_codebook.shape}")
    if hasattr(rrwp_hypernet, "_edges_codebook") and rrwp_hypernet._edges_codebook is not None:
        print(f"  edges_codebook: {rrwp_hypernet._edges_codebook.shape}")
    else:
        n_nodes = rrwp_hypernet.nodes_codebook.shape[0]
        print(f"  edges_codebook: (lazy, estimated {n_nodes}^2 = {n_nodes**2} entries)")

    # -------------------------------------------------------------------
    # 4. Augment samples with RRWP features
    # -------------------------------------------------------------------
    print("\nAugmenting samples with RRWP features...")
    rrwp_samples = []
    for d in tqdm(base_samples, desc="Augmenting"):
        aug = d.clone()
        aug = augment_data_with_rw(
            aug,
            k_values=rw_config.k_values,
            num_bins=rw_config.num_bins,
            bin_boundaries=rw_config.bin_boundaries,
            clip_range=rw_config.clip_range,
        )
        rrwp_samples.append(aug)

    # -------------------------------------------------------------------
    # 5. Run RRWP condition
    # -------------------------------------------------------------------
    rrwp_summary, rrwp_detail = run_condition(
        condition_name="rrwp",
        hypernet=rrwp_hypernet,
        samples=rrwp_samples,
        base_feature_dim=base_feature_dim,
        dataset_name=args.dataset,
        decoder=args.decoder,
        beam_size=args.beam_size,
        skip_graph_decode=args.skip_graph_decode,
        device=device,
    )

    # Build tag and experiment params early so we can save incrementally
    k_str = "_".join(str(k) for k in k_values)
    tag = f"{args.dataset}_dim{args.hv_dim}_depth{depth}_k{k_str}_b{args.num_bins}_{args.decoder}"
    experiment_params = {
        "dataset": args.dataset,
        "hv_dim": args.hv_dim,
        "depth": depth,
        "k_values": args.k_values,
        "num_bins": args.num_bins,
        "decoder": args.decoder,
        "beam_size": args.beam_size,
        "seed": args.seed,
    }

    # Save RRWP results immediately (safe against abort during baseline)
    _save_condition(output_dir, tag, experiment_params, "rrwp", rrwp_summary, rrwp_detail)

    # -------------------------------------------------------------------
    # 6. Run baseline condition
    # -------------------------------------------------------------------
    baseline_summary = None
    baseline_detail = None

    if not args.skip_baseline:
        baseline_config = create_dynamic_config(args.dataset, "HRR", args.hv_dim, depth)
        baseline_hypernet = HyperNet(config=baseline_config, depth=depth).eval().to(device)

        baseline_summary, baseline_detail = run_condition(
            condition_name="baseline",
            hypernet=baseline_hypernet,
            samples=base_samples,
            base_feature_dim=base_feature_dim,
            dataset_name=args.dataset,
            decoder=args.decoder,
            beam_size=args.beam_size,
            skip_graph_decode=args.skip_graph_decode,
            device=device,
        )
        _save_condition(output_dir, tag, experiment_params, "baseline", baseline_summary, baseline_detail)

    # -------------------------------------------------------------------
    # 7. Save combined results
    # -------------------------------------------------------------------
    # JSON summary with both conditions + deltas
    combined = {"args": vars(args), "rrwp": rrwp_summary}
    if baseline_summary is not None:
        combined["baseline"] = baseline_summary
        deltas = {}
        for key in ["edge_accuracy_full", "edge_accuracy_base", "graph_accuracy",
                     "graph_hits_pct", "cosine_similarity"]:
            if key in rrwp_summary and key in baseline_summary:
                deltas[f"delta_{key}"] = rrwp_summary[key] - baseline_summary[key]
        combined["deltas"] = deltas

    with open(output_dir / f"{tag}_summary.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)

    if baseline_detail is not None:
        all_detail = pd.concat([baseline_detail, rrwp_detail], ignore_index=True)
        all_detail.to_csv(output_dir / f"{tag}_all_detailed.csv", index=False)

    # Comparison plots
    for metric, ylabel in [
        ("edge_accuracy_base", "Edge Accuracy (base features)"),
        ("edge_accuracy_full", "Edge Accuracy (full features)"),
    ]:
        plot_comparison(baseline_detail, rrwp_detail, output_dir, metric=metric, ylabel=ylabel, tag=tag)

    if not args.skip_graph_decode:
        plot_comparison(baseline_detail, rrwp_detail, output_dir, metric="graph_accuracy", ylabel="Graph Accuracy", tag=tag)

    print(f"\nResults saved to {output_dir}")
    print(f"  Summary: {tag}_summary.json")
    print(f"  Details: {tag}_*_detailed.csv")


if __name__ == "__main__":
    main()

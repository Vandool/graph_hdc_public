#!/usr/bin/env python
"""
Optuna HPO for Flow Matching (CFM) on ZINC250K with RRWP features.

Compact RRWP config from sweep analysis: dim=512, depth=3, k=(4,8,12), bins=8.
Single objective: minimize best validation loss (MSE).

Post-training evaluation: sample 1000 molecules, decode with greedy beam search,
compute generation metrics (validity, uniqueness, novelty, diversity).

Portable CSV results: run on local machine, copy CSV to cluster, continue.
The Optuna study is rebuilt from CSV if the SQLite database is missing.

Usage:
    # Run 50 trials with post-training evaluation
    python hpo_flow_matching.py --n_trials 50

    # Quick run without evaluation (just optimize val loss)
    python hpo_flow_matching.py --n_trials 10 --no_eval

    # Use full decode_graph pipeline instead of greedy-only
    python hpo_flow_matching.py --n_trials 10 --no_greedy_only
"""

import argparse
import datetime
import json
import math
import os
import random
import string
import tempfile
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import optuna
import pandas as pd
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from graph_hdc import (
    CorrectionLevel,
    DecoderSettings,
    FallbackDecoderSettings,
    GenerationEvaluator,
    HyperNet,
)
from graph_hdc.datasets.utils import (
    get_split,
    post_compute_encodings,
    scan_node_features_with_rw,
)
from graph_hdc.hypernet.configs import RWConfig, create_config_with_rw
from graph_hdc.models.flow_matching import FlowMatchingModel

# ── Environment ──────────────────────────────────────────────────────

# Fix for PyTorch Lightning's _atomic_save which uses tmpfs (limited quota).
_CUSTOM_TMPDIR = Path.cwd() / ".tmp_checkpoints"
_CUSTOM_TMPDIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(_CUSTOM_TMPDIR)
tempfile.tempdir = str(_CUSTOM_TMPDIR)

DTYPE = torch.float32
torch.set_default_dtype(DTYPE)
os.environ.setdefault("PYTHONUNBUFFERED", "1")


# ── Logging ──────────────────────────────────────────────────────────


def log(msg: str) -> None:
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── RRWP Configuration ──────────────────────────────────────────────
# Compact config from sweep: dim=512, depth=3, k=(4,8,12), bins=8

COMPACT_RW_CONFIG = RWConfig(enabled=True, k_values=(4, 8, 12), num_bins=8)
HV_DIM = 512
DATA_DIM = 2 * HV_DIM  # [edge_terms | graph_terms]


def create_zinc_compact_config() -> "DSHDCConfig":
    """Compact RRWP config from sweep analysis report."""
    return create_config_with_rw(
        base_dataset="zinc",
        hv_dim=HV_DIM,
        rw_config=COMPACT_RW_CONFIG,
        hypernet_depth=3,
        prune_codebook=True,
    )


# ── Data Preparation ────────────────────────────────────────────────


def prepare_data(
    device: torch.device,
) -> tuple[list[Data], list[Data], "DSHDCConfig", HyperNet]:
    """
    Load ZINC, encode with HyperNet (RRWP), prepare for FM training.

    The FlowMatchingModel trains on [node_terms | graph_terms] via
    _extract_vectors. We want it to learn [edge_terms | graph_terms],
    so we copy edge_terms into the node_terms slot of each Data object.

    Returns (train_encoded, valid_encoded, config, hypernet).
    """
    config = create_zinc_compact_config()
    config.device = str(device)
    config.dtype = "float32"

    log("Loading ZINC train/valid splits...")
    train_ds = get_split("train", dataset="zinc")
    valid_ds = get_split("valid", dataset="zinc")
    log(f"  Train: {len(train_ds)}, Valid: {len(valid_ds)}")

    # Cache scanned features to avoid re-scanning (~3 min) on every run
    import pickle

    cache_path = RESULTS_DIR / "observed_features_zinc_compact.pkl"
    if cache_path.exists():
        log(f"Loading cached observed features from {cache_path.name}")
        with open(cache_path, "rb") as f:
            observed_features = pickle.load(f)
    else:
        log("Scanning observed node features with RRWP augmentation...")
        observed_features = scan_node_features_with_rw("zinc", COMPACT_RW_CONFIG)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(observed_features, f)
    log(f"  Observed {len(observed_features)} unique feature tuples")

    log("Creating HyperNet (dim=512, depth=3, k=(4,8,12), bins=8)...")
    hypernet = HyperNet(config, observed_node_features=observed_features)
    hypernet.to(device=device, dtype=DTYPE)
    hypernet.eval()

    log("Computing HDC encodings for train set...")
    train_encoded = post_compute_encodings(train_ds, hypernet, device=device)
    log("Computing HDC encodings for valid set...")
    valid_encoded = post_compute_encodings(valid_ds, hypernet, device=device)

    # Swap edge_terms → node_terms so FlowMatchingModel._extract_vectors
    # returns [edge_terms | graph_terms] when vector_part="both".
    # Move every Data object fully to CPU so DataLoader pin_memory works.
    log("Preparing data: setting node_terms = edge_terms, moving to CPU...")
    for d in train_encoded + valid_encoded:
        d.node_terms = d.edge_terms
        # Move ALL tensor attributes to CPU
        for key in d.keys():
            val = d[key]
            if isinstance(val, torch.Tensor) and val.is_cuda:
                d[key] = val.cpu()

    log(f"Data ready: hv_dim={HV_DIM}, data_dim={DATA_DIM}")
    return train_encoded, valid_encoded, config, hypernet


# ── Standardization ─────────────────────────────────────────────────


@torch.no_grad()
def fit_standardization(
    model: FlowMatchingModel, loader: DataLoader, device: torch.device
) -> None:
    """Compute per-feature mean/std for the FM model's input space."""
    data_dim = model.data_dim
    cnt = 0
    sum_vec = torch.zeros(data_dim, device=device)
    sumsq_vec = torch.zeros(data_dim, device=device)

    for batch in loader:
        batch = batch.to(device)
        x = model._extract_vectors(batch)
        cnt += x.shape[0]
        sum_vec += x.sum(dim=0)
        sumsq_vec += (x * x).sum(dim=0)

    mu = sum_vec / cnt
    var = (sumsq_vec / cnt - mu**2).clamp_min_(0)
    sigma = var.sqrt().clamp_min_(1e-6)
    model.set_standardization(mu, sigma)


# ── Folder Naming ────────────────────────────────────────────────────


def _fmt_float(x: float) -> str:
    if x == 0:
        return "0"
    s = f"{x:.6g}"
    if "e" in s:
        base, exp = s.split("e")
        return f"{base}e{int(exp)}"
    return s


def make_run_name(trial_number: int, params: dict) -> str:
    """Compact folder name: fm_zinc_t{N}_h{hidden}_nb{blocks}_lr{lr}_bs{batch}."""
    return (
        f"fm_zinc_t{trial_number}"
        f"_h{params.get('hidden_dim', 0)}"
        f"_nb{params.get('num_blocks', 0)}"
        f"_lr{_fmt_float(params.get('lr', 0))}"
        f"_bs{params.get('batch_size', 0)}"
    )


# ── Experiment Directory ─────────────────────────────────────────────

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "hpo_flow_matching"
HPO_DIR = RESULTS_DIR / "hpo"

STUDY_NAME = "fm_zinc_compact_rrwp_512d3k4_8_12b8"


def setup_trial_dirs(name: str) -> dict[str, Path]:
    """Create directory tree for a single trial."""
    exp_dir = RESULTS_DIR / name
    dirs = {
        "exp_dir": exp_dir,
        "models_dir": exp_dir / "models",
        "evals_dir": exp_dir / "evaluations",
        "logs_dir": exp_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ── Greedy Decoding + Evaluation ────────────────────────────────────


def decode_and_evaluate(
    model: FlowMatchingModel,
    hypernet: HyperNet,
    hv_dim: int,
    n_samples: int,
    device: torch.device,
    greedy_only: bool = True,
) -> dict:
    """
    Sample from the FM model, decode molecules, evaluate generation quality.

    Parameters
    ----------
    model : FlowMatchingModel
        Trained flow model.
    hypernet : HyperNet
        HDC encoder/decoder (with RRWP config).
    hv_dim : int
        Hypervector dimension (512 for compact config).
    n_samples : int
        Number of molecules to sample and evaluate.
    device : torch.device
        Computation device.
    greedy_only : bool
        If True, use greedy beam search decoder directly.
        If False, use the full decode_graph pipeline.
    """
    model.eval()
    model.to(device)

    log(f"Sampling {n_samples} molecules from flow model...")
    with torch.no_grad():
        samples = model.sample(n_samples, device=device).cpu()  # [N, 2*hv_dim]

    # Free FM model to reclaim GPU memory for decoder codebooks
    del model
    torch.cuda.empty_cache()

    # Split into edge_terms and graph_terms
    # (model was trained on [edge_terms | graph_terms])
    # Cast to the HyperNet's VSA tensor type (e.g. HRRTensor) so decoder
    # operations (bind/unbind) work correctly.
    # Use CPU for decoding — greedy beam search is CPU-bound (NetworkX)
    # and the codebook can be large.
    decode_device = hypernet.nodes_codebook.device
    vsa_cls = hypernet.vsa.tensor_class
    edge_terms = samples[:, :hv_dim].to(decode_device).as_subclass(vsa_cls)
    graph_terms = samples[:, hv_dim:].to(decode_device).as_subclass(vsa_cls)

    log(f"Decoding {n_samples} molecules (greedy_only={greedy_only})...")

    # Decoder settings for greedy beam search (ZINC defaults)
    fallback_settings = FallbackDecoderSettings(
        beam_size=32,
        limit=1024,
        top_k=1,
    )

    nx_graphs: list[nx.Graph] = []
    final_flags: list[bool] = []
    sims: list[float] = []
    correction_levels: list[CorrectionLevel] = []

    decode_start = time.time()
    for i in tqdm(range(n_samples), desc="Decoding"):
        try:
            if greedy_only:
                result = hypernet.decode_graph_greedy(
                    edge_term=edge_terms[i],
                    graph_term=graph_terms[i],
                    decoder_settings=fallback_settings,
                )
            else:
                decoder_settings = DecoderSettings.get_default_for("zinc")
                decoder_settings.top_k = 1
                result = hypernet.decode_graph(
                    edge_term=edge_terms[i],
                    graph_term=graph_terms[i],
                    decoder_settings=decoder_settings,
                    fallback_decoder_settings=fallback_settings,
                )

            if result.nx_graphs:
                nx_graphs.append(result.nx_graphs[0])
                final_flags.append(True)
                sims.append(
                    result.cos_similarities[0] if result.cos_similarities else 0.0
                )
                correction_levels.append(result.correction_level)
            else:
                nx_graphs.append(nx.Graph())
                final_flags.append(False)
                sims.append(0.0)
                correction_levels.append(result.correction_level)

        except Exception as e:
            log(f"  Decode failed for sample {i}: {e}")
            nx_graphs.append(nx.Graph())
            final_flags.append(False)
            sims.append(0.0)
            correction_levels.append(CorrectionLevel.FAIL)

    decode_elapsed = time.time() - decode_start
    log(
        f"Decoding done in {decode_elapsed:.1f}s ({decode_elapsed / n_samples:.2f}s/sample)"
    )

    # Evaluate with GenerationEvaluator
    log("Running generation evaluation...")
    evaluator = GenerationEvaluator(base_dataset="zinc", device=device)
    eval_results = evaluator.evaluate(
        n_samples=n_samples,
        samples=nx_graphs,
        final_flags=final_flags,
        sims=sims,
        correction_levels=correction_levels,
    )

    # FCD + KL divergence (reuse from evaluate_generation.py)
    from experiments.scripts.evaluate_generation import compute_fcd, compute_kl_divergence

    mols, valid_flags_eval, _, _ = evaluator.get_mols_valid_flags_sims_and_correction_levels()
    valid_smiles: list[str] = []
    gen_properties: dict[str, list[float]] = {"logp": [], "qed": []}

    if mols is not None:
        from rdkit import Chem

        from graph_hdc.utils.evaluator import rdkit_logp, rdkit_qed

        for mol, valid in zip(mols, valid_flags_eval or [], strict=False):
            if valid and mol is not None:
                try:
                    smiles = Chem.MolToSmiles(mol, canonical=True)
                    valid_smiles.append(smiles)
                    gen_properties["logp"].append(rdkit_logp(mol))
                    gen_properties["qed"].append(rdkit_qed(mol))
                except Exception:
                    pass

    fcd_score = compute_fcd(valid_smiles, evaluator.train_smiles_list)
    if fcd_score is not None:
        log(f"  FCD: {fcd_score:.4f}")

    kl_divergences: dict[str, float] = {}
    for prop_name in ("logp", "qed"):
        kl = compute_kl_divergence(
            gen_properties.get(prop_name, []),
            evaluator.train_properties.get(prop_name, []),
        )
        if kl is not None:
            kl_divergences[prop_name] = kl
    if kl_divergences:
        log(f"  KL divergence: {kl_divergences}")

    # Timing info
    eval_results["decode_time_sec"] = round(decode_elapsed, 2)
    eval_results["decode_time_per_sample_sec"] = round(decode_elapsed / n_samples, 4)
    eval_results["greedy_only"] = greedy_only
    eval_results["fcd"] = fcd_score
    eval_results["kl_divergence"] = kl_divergences
    eval_results["valid_smiles"] = valid_smiles[:100]

    # Correction level distribution
    cl_dist = Counter(
        cl.value if hasattr(cl, "value") else str(cl) for cl in correction_levels
    )
    eval_results["correction_distribution"] = dict(cl_dist)

    return eval_results


# ── Optuna Search Space ─────────────────────────────────────────────


def get_search_space() -> dict[str, optuna.distributions.BaseDistribution]:
    """Optuna parameter space for FlowMatchingModel on ZINC (dim=1024 input)."""
    return {
        "batch_size": optuna.distributions.IntDistribution(128, 512, step=128),
        "lr": optuna.distributions.FloatDistribution(5e-5, 5e-3, log=True),
        "weight_decay": optuna.distributions.FloatDistribution(1e-6, 1e-3, log=True),
        "hidden_dim": optuna.distributions.IntDistribution(512, 2048, step=256),
        "num_blocks": optuna.distributions.IntDistribution(4, 10),
        "time_embed_dim": optuna.distributions.CategoricalDistribution([64, 128, 256]),
        "dropout": optuna.distributions.FloatDistribution(0.0, 0.15),
        "use_ot_coupling": optuna.distributions.CategoricalDistribution([True, False]),
        "warmup_epochs": optuna.distributions.IntDistribution(0, 10),
    }


# ── Optuna CSV Portability ──────────────────────────────────────────


def get_csv_path() -> Path:
    return HPO_DIR / "trials_fm_zinc_compact.csv"


def get_db_path() -> Path:
    return HPO_DIR / "fm_zinc_compact.db"


def load_study() -> optuna.Study:
    """Create or load Optuna study backed by SQLite."""
    db = get_db_path()
    db.parent.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=STUDY_NAME,
        direction="minimize",
        storage=f"sqlite:///{db}",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42, consider_endpoints=True),
    )


def rebuild_study_from_csv() -> optuna.Study:
    """Rebuild Optuna study from portable CSV (e.g. when moving to a new machine)."""
    csv = get_csv_path()
    study = load_study()

    if not csv.exists():
        log("No CSV found; starting fresh study.")
        return study

    df = pd.read_csv(csv)
    if df.empty:
        log("Empty CSV; starting fresh study.")
        return study

    space = get_search_space()
    added = 0

    for _, row in df.iterrows():
        # Reconstruct params from CSV columns
        params: dict = {}
        for k, dist in space.items():
            if k in row and pd.notna(row[k]):
                val = row[k]
                if k == "use_ot_coupling":
                    val = bool(val) if not isinstance(val, bool) else val
                elif k == "time_embed_dim":
                    val = int(val)
                elif isinstance(dist, optuna.distributions.IntDistribution):
                    val = int(val)
                else:
                    val = float(val)
                params[k] = val

        value = row.get("value")
        if pd.isna(value):
            continue

        # Collect user attributes (everything beyond standard + param columns)
        standard_cols = {"number", "value", "state"} | set(space.keys())
        user_attrs: dict = {}
        for col in row.index:
            if col not in standard_cols and pd.notna(row[col]):
                val = row[col]
                if hasattr(val, "item"):
                    val = val.item()
                user_attrs[col] = val

        t = optuna.trial.create_trial(
            params=params,
            distributions=space,
            value=float(value),
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs=user_attrs,
        )
        study.add_trial(t)
        added += 1

    log(f"Rebuilt {added} trials from {csv}")
    return study


def export_trials(study: optuna.Study) -> None:
    """Export all trials to portable CSV file."""
    space = get_search_space()
    rows = []

    for t in study.get_trials(deepcopy=False):
        row: dict = {
            "number": t.number,
            "value": t.value,
            "state": t.state.name if hasattr(t.state, "name") else str(t.state),
        }
        for k in space:
            row[k] = t.params.get(k)
        for attr_name, attr_value in t.user_attrs.items():
            row[attr_name] = attr_value
        rows.append(row)

    csv = get_csv_path()
    csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(csv, index=False)
    log(f"Exported {len(rows)} trials to {csv}")


# ── Single Trial ─────────────────────────────────────────────────────


def run_trial(
    trial: optuna.Trial,
    train_data: list[Data],
    valid_data: list[Data],
    hypernet: HyperNet,
    device: torch.device,
    *,
    epochs: int = 500,
    n_eval_samples: int = 1000,
    greedy_only: bool = True,
    skip_eval: bool = False,
) -> float:
    """
    Train a FlowMatchingModel with Optuna-sampled hyperparameters.

    Returns best validation loss (the Optuna objective).
    """
    # ── Sample hyperparameters ──
    batch_size = trial.suggest_int("batch_size", 128, 512, step=128)
    lr = trial.suggest_float("lr", 5e-5, 5e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    hidden_dim = trial.suggest_int("hidden_dim", 512, 2048, step=256)
    num_blocks = trial.suggest_int("num_blocks", 4, 10)
    time_embed_dim = trial.suggest_categorical("time_embed_dim", [64, 128, 256])
    dropout = trial.suggest_float("dropout", 0.0, 0.15)
    use_ot_coupling = trial.suggest_categorical("use_ot_coupling", [True, False])
    warmup_epochs = trial.suggest_int("warmup_epochs", 0, 10)

    params = {
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "hidden_dim": hidden_dim,
        "num_blocks": num_blocks,
        "time_embed_dim": time_embed_dim,
        "dropout": dropout,
        "use_ot_coupling": use_ot_coupling,
        "warmup_epochs": warmup_epochs,
    }

    run_name = make_run_name(trial.number, params)
    dirs = setup_trial_dirs(run_name)
    log(f"Trial {trial.number}: {run_name}")

    seed_everything(42, workers=True)

    # ── Data loaders ──
    # num_workers=0 avoids CUDA-in-forked-subprocess errors.
    # pin_memory=True only works with CPU tensors; encoded data is on CPU
    # after post_compute_encodings (tensors are .cpu()'d there).
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )

    # ── Model ──
    model = FlowMatchingModel(
        data_dim=DATA_DIM,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        time_embed_dim=time_embed_dim,
        dropout=dropout,
        use_ot_coupling=use_ot_coupling,
        lr=lr,
        weight_decay=weight_decay,
        warmup_epochs=warmup_epochs,
        solver_method="midpoint",
        default_sample_steps=100,
        vector_part="both",
    ).to(device)

    # ── Standardization ──
    fit_standardization(model, train_loader, device)

    # ── Callbacks ──
    ckpt_cb = ModelCheckpoint(
        dirpath=str(dirs["models_dir"]),
        filename="best-{epoch:03d}-{val/loss:.6f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )
    early_stop_cb = EarlyStopping(
        monitor="val/loss",
        mode="min",
        patience=50,
        min_delta=0.0,
        check_finite=True,
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    csv_logger = CSVLogger(str(dirs["logs_dir"]), name="train")

    # ── Precision ──
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        precision = "bf16-mixed"
    elif torch.cuda.is_available():
        precision = "16-mixed"
    else:
        precision = 32

    trainer = Trainer(
        max_epochs=epochs,
        accelerator="auto",
        devices=1,
        precision=precision,
        callbacks=[ckpt_cb, early_stop_cb, lr_monitor],
        logger=csv_logger,
        log_every_n_steps=100,
        gradient_clip_val=1.0,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
    )

    # ── Train ──
    t_start = time.perf_counter()
    trainer.fit(model, train_loader, valid_loader)
    training_time_min = (time.perf_counter() - t_start) / 60

    # ── Best val loss ──
    best_val_loss = float("inf")
    metrics_path = Path(csv_logger.log_dir) / "metrics.csv"
    if metrics_path.exists():
        df = pd.read_csv(metrics_path)
        for col in ("val/loss_epoch", "val/loss"):
            if col in df.columns:
                valid_vals = df[col].dropna()
                if not valid_vals.empty:
                    best_val_loss = float(valid_vals.min())
                    break

    if not math.isfinite(best_val_loss):
        trial.set_user_attr("failure_reason", "DIVERGED")
        return float("inf")

    # ── Store trial metadata ──
    trial.set_user_attr("training_time_min", round(training_time_min, 1))
    trial.set_user_attr("best_val_loss", round(best_val_loss, 8))
    trial.set_user_attr("run_name", run_name)
    trial.set_user_attr("best_ckpt", ckpt_cb.best_model_path or "")
    trial.set_user_attr("stopped_epoch", trainer.current_epoch)

    # Save config to trial dir
    config_dict = {
        "data_dim": DATA_DIM,
        "hv_dim": HV_DIM,
        "rrwp_k_values": list(COMPACT_RW_CONFIG.k_values),
        "rrwp_num_bins": COMPACT_RW_CONFIG.num_bins,
        **params,
    }
    (dirs["evals_dir"] / "trial_config.json").write_text(
        json.dumps(config_dict, indent=2)
    )

    # ── Post-training evaluation ──
    if not skip_eval:
        best_path = ckpt_cb.best_model_path
        if best_path and Path(best_path).exists():
            log(f"Loading best checkpoint: {Path(best_path).name}")
            best_model = FlowMatchingModel.load_from_checkpoint(best_path)
            best_model.to(device).eval()

            eval_results = decode_and_evaluate(
                model=best_model,
                hypernet=hypernet,
                hv_dim=HV_DIM,
                n_samples=n_eval_samples,
                device=device,
                greedy_only=greedy_only,
            )

            # Save full eval results to JSON
            eval_file = dirs["evals_dir"] / "generation_eval.json"
            serializable = {}
            for k, v in eval_results.items():
                if isinstance(v, (dict, list, float, int, str, bool)):
                    serializable[k] = v
            eval_file.write_text(json.dumps(serializable, indent=2, default=str))

            # Store key metrics in Optuna trial
            for metric in (
                "validity",
                "uniqueness",
                "novelty",
                "nuv",
                "internal_diversity_p1",
                "internal_diversity_p2",
                "decode_time_sec",
            ):
                if metric in eval_results:
                    trial.set_user_attr(metric, round(float(eval_results[metric]), 4))

            # Property stats
            for prop in (
                "logp_mean",
                "logp_std",
                "qed_mean",
                "qed_std",
                "sa_score_mean",
                "sa_score_std",
            ):
                if prop in eval_results and not math.isnan(eval_results[prop]):
                    trial.set_user_attr(prop, round(float(eval_results[prop]), 4))

            # FCD and KL divergence
            if eval_results.get("fcd") is not None:
                trial.set_user_attr("fcd", round(float(eval_results["fcd"]), 4))
            kl = eval_results.get("kl_divergence", {})
            for prop_name, kl_val in kl.items():
                trial.set_user_attr(f"kl_{prop_name}", round(float(kl_val), 4))

            # Correction distribution
            if "correction_distribution" in eval_results:
                for level, count in eval_results["correction_distribution"].items():
                    trial.set_user_attr(f"cl_{level.replace(' ', '_')}", count)

            log(
                f"  Eval: validity={eval_results.get('validity', 0):.1f}%, "
                f"uniqueness={eval_results.get('uniqueness', 0):.1f}%, "
                f"novelty={eval_results.get('novelty', 0):.1f}%, "
                f"nuv={eval_results.get('nuv', 0):.1f}%"
            )

            del best_model
            torch.cuda.empty_cache()

    log(
        f"Trial {trial.number} done: val_loss={best_val_loss:.6f}, time={training_time_min:.1f}min"
    )
    return best_val_loss


# ── Main ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Optuna HPO for Flow Matching on ZINC250K with RRWP features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--n_trials", type=int, default=1, help="Number of Optuna trials to run"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Max training epochs per trial (early stopping patience=50)",
    )
    parser.add_argument(
        "--n_eval_samples",
        type=int,
        default=1000,
        help="Number of molecules to sample for post-training evaluation",
    )
    parser.add_argument(
        "--greedy_only",
        action="store_true",
        default=True,
        help="Use greedy beam search decoder only (default)",
    )
    parser.add_argument(
        "--no_greedy_only",
        dest="greedy_only",
        action="store_false",
        help="Use full decode_graph pipeline instead of greedy-only",
    )
    parser.add_argument(
        "--no_eval",
        action="store_true",
        default=False,
        help="Skip post-training evaluation (faster HPO)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="auto", help="Device: auto, cpu, cuda"
    )
    args = parser.parse_args()

    seed_everything(args.seed, workers=True)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    log(f"Device: {device}")

    # Prepare data (once, reused across all trials)
    train_data, valid_data, config, hypernet = prepare_data(device)

    # Setup Optuna study (rebuild from CSV if DB missing)
    HPO_DIR.mkdir(parents=True, exist_ok=True)
    db_path = get_db_path()

    if not db_path.exists():
        study = rebuild_study_from_csv()
    else:
        study = load_study()

    log(f"Study: {STUDY_NAME}")
    log(f"DB: {db_path}")
    log(f"CSV: {get_csv_path()}")
    log(f"Existing trials: {len(study.trials)}")
    log(f"New trials: {args.n_trials}")
    log(f"Config: dim={HV_DIM}, depth=3, k=(4,8,12), bins=8")
    log(f"Greedy-only: {args.greedy_only}, Skip eval: {args.no_eval}")
    log(f"Eval samples: {args.n_eval_samples}")
    print()

    # Objective wrapper with error handling
    def objective(trial: optuna.Trial) -> float:
        try:
            return run_trial(
                trial=trial,
                train_data=train_data,
                valid_data=valid_data,
                hypernet=hypernet,
                device=device,
                epochs=args.epochs,
                n_eval_samples=args.n_eval_samples,
                greedy_only=args.greedy_only,
                skip_eval=args.no_eval,
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                log(f"Trial {trial.number} CUDA OOM")
                trial.set_user_attr("failure_reason", "CUDA_OOM")
                torch.cuda.empty_cache()
                return float("inf")
            log(f"Trial {trial.number} RuntimeError: {e}")
            trial.set_user_attr("failure_reason", f"RuntimeError: {str(e)[:200]}")
            return float("inf")
        except Exception as e:
            log(f"Trial {trial.number} error: {type(e).__name__}: {e}")
            trial.set_user_attr("failure_reason", f"{type(e).__name__}: {str(e)[:200]}")
            return float("inf")

    # Run optimization
    try:
        log(f"Starting optimization with {args.n_trials} trials...")
        study.optimize(objective, n_trials=args.n_trials)
    except KeyboardInterrupt:
        log("Interrupted by user (Ctrl+C)")
    finally:
        # Always export, even if interrupted
        export_trials(study)

    # ── Summary ──
    print()
    print("=" * 70)
    print("HPO SUMMARY")
    print("=" * 70)
    print(f"Total trials: {len(study.trials)}")

    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
        and t.value is not None
        and math.isfinite(t.value)
    ]

    if study.best_trial and math.isfinite(study.best_value):
        bt = study.best_trial
        print(f"\nBest trial: #{bt.number}")
        print(f"  Val loss: {study.best_value:.6f}")
        if "run_name" in bt.user_attrs:
            print(f"  Run: {bt.user_attrs['run_name']}")
        if "validity" in bt.user_attrs:
            print(f"  Validity:   {bt.user_attrs['validity']:.1f}%")
            print(f"  Uniqueness: {bt.user_attrs.get('uniqueness', 0):.1f}%")
            print(f"  Novelty:    {bt.user_attrs.get('novelty', 0):.1f}%")
            print(f"  NUV:        {bt.user_attrs.get('nuv', 0):.1f}%")
        if "training_time_min" in bt.user_attrs:
            print(f"  Time: {bt.user_attrs['training_time_min']:.1f} min")

        print("\n  Hyperparameters:")
        for k, v in bt.params.items():
            print(f"    {k}: {v}")

    if len(completed) > 1:
        sorted_trials = sorted(completed, key=lambda t: t.value)
        print(f"\nTop 5 trials (by val loss):")
        for i, t in enumerate(sorted_trials[:5], 1):
            validity = t.user_attrs.get("validity", "N/A")
            name = t.user_attrs.get("run_name", f"trial_{t.number}")
            if isinstance(validity, float):
                print(
                    f"  {i}. #{t.number}: val_loss={t.value:.6f}, validity={validity:.1f}%, run={name}"
                )
            else:
                print(
                    f"  {i}. #{t.number}: val_loss={t.value:.6f}, validity={validity}, run={name}"
                )

    print("=" * 70)
    print(f"CSV results: {get_csv_path()}")
    print(f"Trial dirs:  {RESULTS_DIR}")


if __name__ == "__main__":
    main()

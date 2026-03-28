"""
PubChem SMILES Datasets (pubchem16, pubchem32, pubchem64).

PyG InMemoryDataset for PubChem molecules filtered by maximum heavy atom count.

Variants:
  pubchem16 — molecules with ≤16 heavy atoms  (~16 M molecules)
  pubchem32 — molecules with ≤32 heavy atoms  (~94 M molecules, superset of 16)
  pubchem64 — molecules with ≤64 heavy atoms  (~118 M molecules, superset of 32)

Node features: [atom_type, degree-1, formal_charge, total_Hs, is_in_ring]
- Atom types: 12 (B, Br, C, Cl, F, I, N, O, P, S, Si, Sn)
- Degrees: 5 values (0-4, i.e. degree 1-5)
- Formal charges: 3 values (0, 1, 2 for 0, +, -)
- Total Hs: 4 values (0-3)
- Is in ring: 2 values (0, 1)

Combinatorial space: 12 * 5 * 3 * 4 * 2 = 1,440 possible node types

Hydrogen is NOT included as a node type — it is captured implicitly via
the ``total_Hs`` feature.  Molecules with explicit H atoms in the SMILES
are skipped during processing.

Split strategy: deterministic SHA-256 hash of the CID gives 80 / 10 / 10
train / valid / test.  For datasets with millions of molecules this is
statistically equivalent to stratified splitting (each size bin gets the
same 80/10/10 ratio by the law of large numbers).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen
from torch_geometric.data import Data, InMemoryDataset
from tqdm.auto import tqdm

from graph_hdc.utils.helpers import ROOT

DATASET_PATH = ROOT / "data"

PubChemVariant = Literal["pubchem16", "pubchem32", "pubchem64"]

PUBCHEM_ATOM_TO_IDX: dict[str, int] = {
    "B": 0,
    "Br": 1,
    "C": 2,
    "Cl": 3,
    "F": 4,
    "I": 5,
    "N": 6,
    "O": 7,
    "P": 8,
    "S": 9,
    "Si": 10,
    "Sn": 11,
}
PUBCHEM_IDX_TO_ATOM: dict[int, str] = {v: k for k, v in PUBCHEM_ATOM_TO_IDX.items()}

_VARIANT_INFO: dict[PubChemVariant, dict] = {
    "pubchem16": {"csv": "pubchem16.csv", "max_heavy": 16, "dir": "PubChem16Smiles"},
    "pubchem32": {"csv": "pubchem32.csv", "max_heavy": 32, "dir": "PubChem32Smiles"},
    "pubchem64": {"csv": "pubchem64.csv", "max_heavy": 64, "dir": "PubChem64Smiles"},
}


def _cid_to_split(cid: int) -> str:
    """Deterministic split assignment using SHA-256 hash of CID.

    For large datasets (millions of molecules) this is statistically
    equivalent to stratified splitting: every size bin receives the same
    80 / 10 / 10 ratio by the law of large numbers, because CID order
    is independent of molecular properties.
    """
    h = int(hashlib.sha256(str(cid).encode()).hexdigest()[:8], 16) % 1000
    if h < 800:
        return "train"
    elif h < 900:
        return "valid"
    else:
        return "test"


def mol_to_data(mol: Chem.Mol) -> Data:
    """Convert RDKit molecule to PyG Data with PubChem features."""
    x = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in PUBCHEM_ATOM_TO_IDX:
            raise ValueError(f"Unexpected atom '{sym}' for PubChem.")
        x.append([
            float(PUBCHEM_ATOM_TO_IDX[sym]),
            float(min(4, max(0, atom.GetDegree() - 1))),
            float(min(2, atom.GetFormalCharge()) if atom.GetFormalCharge() >= 0 else 2),
            float(min(3, atom.GetTotalNumHs())),
            float(atom.IsInRing()),
        ])

    src, dst = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        src += [i, j]
        dst += [j, i]

    logp = Crippen.MolLogP(mol)
    qed_val = QED.qed(mol)

    data_dict = {
        "x": torch.tensor(x, dtype=torch.float32),
        "edge_index": torch.tensor([src, dst], dtype=torch.long),
        "smiles": Chem.MolToSmiles(mol, canonical=True),
        "logp": torch.tensor([float(logp)], dtype=torch.float32),
        "qed": torch.tensor([float(qed_val)], dtype=torch.float32),
    }

    return Data(**data_dict)


class PubChemSmiles(InMemoryDataset):
    """
    PubChem SMILES dataset parameterised by molecule size.

    Parameters
    ----------
    variant : {"pubchem16", "pubchem32", "pubchem64"}
        Which size variant to load.
    split : {"train", "valid", "test"}
        Dataset split.
    max_molecules : int or None
        Cap on the number of molecules for the *train* split.  Valid and
        test splits are scaled to ``max_molecules * 0.1`` (valid) and
        ``max_molecules * 0.05`` (test).  Defaults
        per variant:
        pubchem16 → 50 000, pubchem32 → 50 000, pubchem64 → 50 000.
        Pass ``None`` to load everything (requires substantial RAM).
    """

    _SPLIT_SCALE = {"train": 1.0, "valid": 0.1, "test": 0.05}

    _DEFAULT_MAX_MOLECULES: dict[str, int] = {
        "pubchem16": 250_000,
        "pubchem32": 250_000,
        "pubchem64": 250_000,
    }

    def __init__(
        self,
        variant: PubChemVariant = "pubchem16",
        split: str = "train",
        max_molecules: int | None = ...,  # sentinel — use default per variant
        root: str | Path | None = None,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ) -> None:
        self.variant = variant
        self.split = split.lower()
        assert self.variant in _VARIANT_INFO, f"Unknown variant: {variant}"
        assert self.split in {"train", "valid", "test"}

        # Resolve default cap
        if max_molecules is ...:
            max_molecules = self._DEFAULT_MAX_MOLECULES[variant]

        if max_molecules is not None:
            self.max_molecules = max(1, int(max_molecules * self._SPLIT_SCALE[self.split]))
        else:
            self.max_molecules = None

        if root is None:
            root = DATASET_PATH / _VARIANT_INFO[variant]["dir"]

        super().__init__(str(root), transform, pre_transform, pre_filter)

        with open(self.processed_paths[0], "rb") as f:
            self.data, self.slices = torch.load(
                f, map_location="cpu", weights_only=False,
            )

    @property
    def raw_file_names(self) -> list[str]:
        return [_VARIANT_INFO[self.variant]["csv"]]

    @property
    def processed_file_names(self) -> list[str]:
        cap = f"_max{self.max_molecules}" if self.max_molecules is not None else ""
        return [f"data_{self.split}{cap}.pt"]

    def download(self):
        """Symlink the CSV from data/pubchem/ if not already present."""
        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        csv_name = _VARIANT_INFO[self.variant]["csv"]
        target = raw_dir / csv_name
        if not target.exists():
            source = DATASET_PATH / "pubchem" / csv_name
            if source.exists():
                target.symlink_to(source)
            else:
                raise FileNotFoundError(
                    f"Cannot find {source}. Place {csv_name} in {DATASET_PATH / 'pubchem'}."
                )

    def process(self):
        csv_path = Path(self.raw_paths[0])
        data_list: list[Data] = []
        skipped_disconnected = 0
        skipped_atoms = 0
        skipped_parse = 0

        reader = pd.read_csv(csv_path, chunksize=50_000)
        for chunk in tqdm(reader, desc=f"{self.variant}[{self.split}] reading CSV"):
            for _, row in chunk.iterrows():
                cid = int(row["CID"])
                smi = str(row["SMILES"])

                # Deterministic split by CID hash
                if _cid_to_split(cid) != self.split:
                    continue

                # Skip disconnected molecules
                if "." in smi:
                    skipped_disconnected += 1
                    continue

                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    skipped_parse += 1
                    continue

                # Check all atoms are supported
                if any(a.GetSymbol() not in PUBCHEM_ATOM_TO_IDX for a in mol.GetAtoms()):
                    skipped_atoms += 1
                    continue

                data = mol_to_data(mol)
                if self.pre_filter and not self.pre_filter(data):
                    continue
                if self.pre_transform:
                    data = self.pre_transform(data)
                data_list.append(data)

                if self.max_molecules is not None and len(data_list) >= self.max_molecules:
                    break

            if self.max_molecules is not None and len(data_list) >= self.max_molecules:
                break

        print(f"{self.variant}[{self.split}]: {len(data_list)} molecules processed")
        print(f"  skipped: {skipped_disconnected} disconnected, "
              f"{skipped_parse} unparseable, {skipped_atoms} unsupported atoms")

        data, slices = self.collate(data_list)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

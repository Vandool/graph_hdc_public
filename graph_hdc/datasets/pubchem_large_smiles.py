"""
PubChem Large Molecules SMILES Dataset.

PyG InMemoryDataset for large PubChem molecules (32-64 heavy atoms) from a CSV.

Node features: [atom_type, degree-1, formal_charge, total_Hs, is_in_ring]
- Atom types: 13 (B, Br, C, Cl, F, H, I, N, O, P, S, Si, Sn)
- Degrees: 5 values (0-4, i.e. degree 1-5)
- Formal charges: 3 values (0, 1, 2 for 0, +, -)
- Total Hs: 4 values (0-3)
- Is in ring: 2 values (0, 1)

Combinatorial space: 13 * 5 * 3 * 4 * 2 = 1,560 possible node types

The source CSV has columns (CID, SMILES).  Splits are determined
deterministically by ``CID % 10``:
  train: CID % 10 < 8   (80 %)
  valid: CID % 10 == 8   (10 %)
  test:  CID % 10 == 9   (10 %)

Because the full CSV contains ~24 M molecules, an optional *max_molecules*
cap (default 50 000 per split) keeps processing time and memory practical.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import QED, Crippen
from torch_geometric.data import Data, InMemoryDataset
from tqdm.auto import tqdm

from graph_hdc.utils.helpers import ROOT

DATASET_PATH = ROOT / "data"

PUBCHEM_LARGE_ATOM_TO_IDX: dict[str, int] = {
    "B": 0,
    "Br": 1,
    "C": 2,
    "Cl": 3,
    "F": 4,
    "H": 5,
    "I": 6,
    "N": 7,
    "O": 8,
    "P": 9,
    "S": 10,
    "Si": 11,
    "Sn": 12,
}
PUBCHEM_LARGE_IDX_TO_ATOM: dict[int, str] = {v: k for k, v in PUBCHEM_LARGE_ATOM_TO_IDX.items()}

# Split mapping: CID % 10 -> split name
_SPLIT_MAP = {i: "train" for i in range(8)}
_SPLIT_MAP[8] = "valid"
_SPLIT_MAP[9] = "test"


def mol_to_data(mol: Chem.Mol) -> Data:
    """Convert RDKit molecule to PyG Data with PubChem-Large features."""
    x = []
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym not in PUBCHEM_LARGE_ATOM_TO_IDX:
            raise ValueError(f"Unexpected atom '{sym}' for PubChem-Large.")
        x.append([
            float(PUBCHEM_LARGE_ATOM_TO_IDX[sym]),
            float(min(4, max(0, atom.GetDegree() - 1))),
            float(atom.GetFormalCharge() if atom.GetFormalCharge() >= 0 else 2),
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


class PubChemLargeSmiles(InMemoryDataset):
    """
    PubChem large-molecule SMILES dataset.

    Reads ``pubchem_large_only.csv`` and splits deterministically by CID.

    Parameters
    ----------
    root : Path
        Dataset root directory.
    split : str
        One of {"train", "valid", "test"}.
    max_molecules : int or None
        Maximum molecules for the *train* split.  Valid and test splits
        are scaled to 1/8 of this value (preserving the 80/10/10 ratio).
        ``None`` means no limit.
    """

    # Split ratios: train gets 8/10, valid 1/10, test 1/10 of CIDs.
    # Scale max_molecules accordingly so cached sizes reflect the ratio.
    _SPLIT_SCALE = {"train": 1.0, "valid": 0.125, "test": 0.125}

    def __init__(
        self,
        root: str | Path = DATASET_PATH / "PubChemLargeSmiles",
        split: str = "train",
        max_molecules: int | None = 50_000,
        transform: Callable | None = None,
        pre_transform: Callable | None = None,
        pre_filter: Callable | None = None,
    ) -> None:
        self.split = split.lower()
        if max_molecules is not None:
            self.max_molecules = max(1, int(max_molecules * self._SPLIT_SCALE[self.split]))
        else:
            self.max_molecules = None
        assert self.split in {"train", "valid", "test"}
        super().__init__(root, transform, pre_transform, pre_filter)

        with open(self.processed_paths[0], "rb") as f:
            self.data, self.slices = torch.load(f, map_location="cpu", weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        return ["pubchem_large_only.csv"]

    @property
    def processed_file_names(self) -> list[str]:
        cap = f"_max{self.max_molecules}" if self.max_molecules is not None else ""
        return [f"data_{self.split}{cap}.pt"]

    def download(self):
        """Symlink the CSV from the data/ directory if not already present."""
        raw_dir = Path(self.raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / "pubchem_large_only.csv"
        if not target.exists():
            source = DATASET_PATH / "pubchem_large_only.csv"
            if source.exists():
                target.symlink_to(source)
            else:
                raise FileNotFoundError(
                    f"Cannot find {source}. Place pubchem_large_only.csv in {DATASET_PATH}."
                )

    def process(self):
        csv_path = Path(self.raw_paths[0])
        data_list: list[Data] = []
        skipped_disconnected = 0
        skipped_atoms = 0
        skipped_parse = 0

        reader = pd.read_csv(csv_path, chunksize=50_000)
        for chunk in tqdm(reader, desc=f"PubChemLarge[{self.split}] reading CSV"):
            for _, row in chunk.iterrows():
                cid = int(row["CID"])
                smi = str(row["SMILES"])

                # Deterministic split by CID
                if _SPLIT_MAP[cid % 10] != self.split:
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
                if any(a.GetSymbol() not in PUBCHEM_LARGE_ATOM_TO_IDX for a in mol.GetAtoms()):
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

        print(f"PubChemLarge[{self.split}]: {len(data_list)} molecules processed")
        print(f"  skipped: {skipped_disconnected} disconnected, "
              f"{skipped_parse} unparseable, {skipped_atoms} unsupported atoms")

        data, slices = self.collate(data_list)
        Path(self.processed_dir).mkdir(parents=True, exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])

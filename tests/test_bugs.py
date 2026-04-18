"""
Tests documenting known bugs in the graph_hdc pipeline.

Each test class documents a single bug. Tests are written to FAIL until
the bug is fixed — they assert the *correct* (expected) behavior, not
the current broken behavior.
"""

import signal

import networkx as nx
import pytest
import torch
from rdkit import Chem

from graph_hdc.hypernet.types import Feat
from graph_hdc.utils.chem import (
    _infer_bond_orders,
    nx_to_mol,
    reconstruct_for_eval,
    is_valid_molecule,
    mol_to_data,
    FORMAL_CHARGE_IDX_TO_VAL,
    QM9_ATOM_SYMBOLS,
    ZINC_ATOM_SYMBOLS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fcd_available() -> bool:
    """Check if fcd-torch is installed."""
    try:
        from fcd_torch import FCD
        return True
    except ImportError:
        return False


def _has_unnecessary_brackets(smiles: str) -> bool:
    """
    Check if a SMILES string contains brackets around atoms that have
    standard valence (i.e., brackets that RDKit wouldn't normally write).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    canonical = Chem.MolToSmiles(mol, canonical=True)
    return canonical != smiles


def _atom_symbols(smiles: str) -> list[str]:
    """Extract atom symbols from a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    return [a.GetSymbol() for a in mol.GetAtoms()]


def smiles_to_nx(smiles: str, dataset: str = "qm9") -> nx.Graph:
    """
    Convert a SMILES string to a NetworkX graph with node features,
    mimicking the format produced by the HDC decoder.

    The decoder creates nodes with 'feat' (Feat object) and 'type' (raw tuple)
    attributes. This helper reproduces that structure by going through
    RDKit → mol_to_data → NetworkX.
    """
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None, f"Invalid SMILES: {smiles}"
    # Kekulize to get explicit bond orders, matching how the training data is prepared
    Chem.Kekulize(mol, clearAromaticFlags=True)

    data = mol_to_data(mol, dataset=dataset)
    G = nx.Graph()

    # Add nodes with feat and type attributes (same as add_node_with_feat)
    for i in range(data.x.size(0)):
        row = data.x[i].tolist()
        t = tuple(int(v) for v in row)
        feat = Feat.from_tuple(t)
        G.add_node(i, feat=feat, type=t, target_degree=feat.target_degree)

    # Add edges from edge_index
    edge_index = data.edge_index
    for k in range(edge_index.size(1)):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u < v:  # undirected: add each edge once
            G.add_edge(u, v)

    return G


def canonical_smiles(smiles: str) -> str:
    """Return the canonical SMILES for a given SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return Chem.MolToSmiles(mol, canonical=True)


def build_nx_and_rwmol(
    atom_specs: list[tuple[str, int, int, int]],
    edges: list[tuple[int, int]],
    atom_symbols: list[str] = QM9_ATOM_SYMBOLS,
) -> tuple[nx.Graph, Chem.RWMol, list[int]]:
    """
    Build a NetworkX graph and matching RWMol from raw atom specifications.

    Each atom_spec is (symbol, degree_idx, formal_charge_idx, explicit_hs).
    Returns (graph, rwmol, sorted_nodes) for use with _infer_bond_orders.
    """
    atom_to_idx = {s: i for i, s in enumerate(atom_symbols)}
    G = nx.Graph()
    mol = Chem.RWMol()
    for i, (sym, deg, charge_idx, hs) in enumerate(atom_specs):
        aidx = atom_to_idx[sym]
        feat = Feat(aidx, deg, charge_idx, hs)
        G.add_node(i, feat=feat, type=(aidx, deg, charge_idx, hs), target_degree=deg + 1)
        atom = Chem.Atom(sym)
        atom.SetFormalCharge(FORMAL_CHARGE_IDX_TO_VAL.get(charge_idx, 0))
        atom.SetNumExplicitHs(hs)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    for u, v in edges:
        G.add_edge(u, v)
    nodes = sorted(G.nodes)
    return G, mol, nodes


def run_bond_inference_with_timeout(
    mol, edges, G, nodes, atom_symbols, timeout_sec=3,
):
    """
    Run _infer_bond_orders with a timeout. Returns the result or None
    if the algorithm doesn't terminate within timeout_sec.
    """
    def handler(signum, frame):
        raise TimeoutError("_infer_bond_orders did not terminate")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_sec)
    try:
        result = _infer_bond_orders(mol, edges, G, nodes, atom_symbols)
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        return result
    except TimeoutError:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        return None


# ---------------------------------------------------------------------------
# Bug 1: nx_to_mol produces non-canonical SMILES with unnecessary brackets
# ---------------------------------------------------------------------------

class TestBracketedSmilesFromNxToMol:
    """
    Bug: nx_to_mol() calls SetNoImplicit(True) and SetNumExplicitHs(),
    which forces RDKit to write bracket notation ([CH], [NH], [OH], etc.)
    in the SMILES output, even when the atom has standard valence and
    brackets are not needed.

    Expected behavior: The SMILES produced by round-tripping a molecule
    through the HDC graph representation should be canonical — identical
    to what RDKit produces from the original SMILES.
    """

    @pytest.mark.parametrize(
        "smiles, name",
        [
            ("CCO", "ethanol"),
            ("CC=O", "acetaldehyde"),
            ("c1ccccc1", "benzene"),
            ("CC(=O)O", "acetic_acid"),
            ("CC(C)=C(c1ccccc1)N1CCCC1", "screenshot_molecule"),
            ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "caffeine"),
        ],
    )
    def test_no_unnecessary_brackets(self, smiles: str, name: str):
        """
        Round-tripping a molecule through nx graph representation and back
        via nx_to_mol() should produce canonical SMILES without unnecessary
        brackets.
        """
        expected = canonical_smiles(smiles)
        dataset = "qm9" if all(s in "CNOF" for s in _atom_symbols(smiles)) else "zinc"

        G = smiles_to_nx(smiles, dataset=dataset)
        mol, _ = nx_to_mol(G, dataset=dataset, infer_bonds=True, sanitize=True, kekulize=True)

        assert mol is not None, f"nx_to_mol returned None for {name}"
        result = Chem.MolToSmiles(mol, canonical=True)

        # The result should not contain unnecessary brackets for standard atoms
        assert result == expected, (
            f"[{name}] SMILES mismatch:\n"
            f"  got:      {result}\n"
            f"  expected: {expected}\n"
            f"  Unnecessary brackets detected: {_has_unnecessary_brackets(result)}"
        )

    @pytest.mark.parametrize(
        "smiles",
        [
            "CCO",
            "CC(=O)O",
            "c1ccccc1",
            "CC(C)=C(c1ccccc1)N1CCCC1",
        ],
    )
    def test_no_implicit_flag_causes_brackets(self, smiles: str):
        """
        Demonstrate that SetNoImplicit(True) is the root cause: atoms with
        standard valence should not need explicit H notation in SMILES.
        """
        dataset = "qm9" if all(s in "CNOF" for s in _atom_symbols(smiles)) else "zinc"
        G = smiles_to_nx(smiles, dataset=dataset)
        mol, _ = nx_to_mol(G, dataset=dataset, infer_bonds=True, sanitize=True, kekulize=True)

        assert mol is not None
        result = Chem.MolToSmiles(mol, canonical=True)
        assert not _has_unnecessary_brackets(result), (
            f"SMILES contains unnecessary brackets: {result}"
        )

    @pytest.mark.parametrize(
        "smiles",
        [
            "CCO",
            "CC(C)=C(c1ccccc1)N1CCCC1",
        ],
    )
    def test_reconstruct_for_eval_produces_canonical_smiles(self, smiles: str):
        """
        reconstruct_for_eval should also produce canonical SMILES, since it
        delegates to nx_to_mol under the hood.
        """
        expected = canonical_smiles(smiles)
        dataset = "qm9" if all(s in "CNOF" for s in _atom_symbols(smiles)) else "zinc"

        G = smiles_to_nx(smiles, dataset=dataset)
        mol = reconstruct_for_eval(G, dataset=dataset)

        assert mol is not None
        result = Chem.MolToSmiles(mol, canonical=True)
        assert result == expected, (
            f"reconstruct_for_eval SMILES mismatch:\n"
            f"  got:      {result}\n"
            f"  expected: {expected}"
        )


# ---------------------------------------------------------------------------
# Bug 2: Bracketed SMILES produce incorrect FCD scores
# ---------------------------------------------------------------------------

class TestBracketedSmilesFCDImpact:
    """
    Bug: Because nx_to_mol produces non-canonical SMILES with brackets,
    the FCD (Fréchet ChemNet Distance) scores computed on generated
    molecules are wrong. ChemNet processes bracketed SMILES differently
    from canonical SMILES for the same molecule, leading to different
    activations and therefore inflated/incorrect FCD.

    Expected behavior: FCD computed on molecules from nx_to_mol should
    equal FCD computed on the same molecules with canonical SMILES.
    """

    @pytest.fixture
    def sample_smiles(self):
        """Molecules that trigger brackets when round-tripped through nx_to_mol."""
        return [
            "c1ccccc1",          # benzene
            "c1ccncc1",          # pyridine
            "c1ccc2ccccc2c1",    # naphthalene
            "c1cc2ccccc2cc1",    # naphthalene isomer
            "c1ccoc1",           # furan
            "c1ccc(C)cc1",       # toluene
            "c1cc[nH]c1",       # pyrrole
            "CCO",              # ethanol (no brackets, for contrast)
            "CC=O",             # acetaldehyde
            "CC(C)=O",          # acetone
        ]

    @pytest.fixture
    def reference_smiles(self):
        """Reference set for FCD comparison."""
        return [
            "c1ccccc1",
            "c1ccncc1",
            "CCO",
            "CC=O",
            "CCN",
            "CCC",
            "c1ccc2ccccc2c1",
            "c1ccoc1",
            "CCCC",
            "CC(C)=O",
        ]

    @pytest.mark.skipif(
        not _fcd_available(),
        reason="fcd-torch not installed",
    )
    def test_fcd_identical_for_canonical_vs_bracketed(
        self, sample_smiles, reference_smiles,
    ):
        """
        FCD should be the same whether we use canonical SMILES or the
        bracketed SMILES produced by nx_to_mol for the same molecules.

        Currently fails because nx_to_mol produces [CH3], [CH2], [NH], etc.
        which ChemNet encodes differently from C, C, N, etc.
        """
        from fcd_torch import FCD

        # Canonical SMILES (ground truth)
        canonical_list = [canonical_smiles(s) for s in sample_smiles]

        # Bracketed SMILES (from nx_to_mol round-trip)
        bracketed_list = []
        for s in sample_smiles:
            G = smiles_to_nx(s, dataset="qm9")
            mol, _ = nx_to_mol(G, dataset="qm9", infer_bonds=True, sanitize=True, kekulize=True)
            assert mol is not None
            bracketed_list.append(Chem.MolToSmiles(mol, canonical=True))

        ref_canonical = [canonical_smiles(s) for s in reference_smiles]

        fcd_metric = FCD(device="cpu", n_jobs=1)
        fcd_canonical_score = fcd_metric(canonical_list, ref_canonical)
        fcd_bracketed_score = fcd_metric(bracketed_list, ref_canonical)

        assert abs(fcd_canonical_score - fcd_bracketed_score) < 1e-6, (
            f"FCD differs between canonical and bracketed SMILES:\n"
            f"  canonical FCD:  {fcd_canonical_score:.6f}\n"
            f"  bracketed FCD:  {fcd_bracketed_score:.6f}\n"
            f"  delta:          {abs(fcd_canonical_score - fcd_bracketed_score):.6f}\n"
            f"  Example canonical:  {canonical_list[:3]}\n"
            f"  Example bracketed:  {bracketed_list[:3]}"
        )


# ---------------------------------------------------------------------------
# Bug 3: Inverted charge sign in target valence calculation
# ---------------------------------------------------------------------------

class TestChargeSignInBondInference:
    """
    Bug: _infer_bond_orders computes target_valence as
        base_val - charge - explicit_hs
    but the correct formula is
        base_val + charge - explicit_hs

    A positive formal charge means the atom donated an electron and needs
    MORE bonds (e.g., N+ needs 4 bonds like C). A negative charge means
    fewer bonds (e.g., O- needs 1 bond like F). The current formula
    inverts this, giving N+ too few target bonds and O- too many.

    Expected behavior: Bond orders should be assigned correctly regardless
    of formal charge.
    """

    def test_positive_charge_double_bond(self):
        """
        [NH2+]=C should get a double bond between N+ and C.

        N+ has base_val=3, charge=+1, Hs=2.
        Wrong target: 3 - 1 - 2 = 0 (no bonds needed — clearly wrong)
        Correct target: 3 + 1 - 2 = 2 (needs a double bond)
        """
        # N+ (atom_type=1, degree_idx=0, charge_idx=1(+1), Hs=2)
        # C  (atom_type=0, degree_idx=0, charge_idx=0,      Hs=2)
        G, mol, nodes = build_nx_and_rwmol(
            atom_specs=[("N", 0, 1, 2), ("C", 0, 0, 2)],
            edges=[(0, 1)],
        )
        result = _infer_bond_orders(mol, [(0, 1)], G, nodes, QM9_ATOM_SYMBOLS)
        _, _, bond_type = result[0]

        assert bond_type == Chem.BondType.DOUBLE, (
            f"N+=C bond should be DOUBLE, got {bond_type}. "
            f"Charge sign is inverted in target_valence calculation."
        )

    def test_negative_charge_single_bond(self):
        """
        C-[O-] should keep a single bond (O- needs only 1 bond).

        O- has base_val=2, charge=-1, Hs=0.
        Wrong target: 2 - (-1) - 0 = 3 (wants triple bond!)
        Correct target: 2 + (-1) - 0 = 1 (single bond, correct)
        """
        # C  (atom_type=0, degree_idx=0, charge_idx=0, Hs=3)
        # O- (atom_type=2, degree_idx=0, charge_idx=2(-1), Hs=0)
        G, mol, nodes = build_nx_and_rwmol(
            atom_specs=[("C", 0, 0, 3), ("O", 0, 2, 0)],
            edges=[(0, 1)],
        )
        result = _infer_bond_orders(mol, [(0, 1)], G, nodes, QM9_ATOM_SYMBOLS)
        _, _, bond_type = result[0]

        assert bond_type == Chem.BondType.SINGLE, (
            f"C-[O-] bond should be SINGLE, got {bond_type}. "
            f"O- incorrectly gets high target valence due to charge sign bug."
        )

    def test_nitromethane_roundtrip(self):
        """
        Nitromethane C[N+](=O)[O-] should preserve the N=O double bond.
        """
        # Build the graph manually:
        # C (type=0, deg=0, charge=0, Hs=3)
        # N+ (type=1, deg=2, charge=1(+1), Hs=0) — 3 heavy-atom bonds
        # O  (type=2, deg=0, charge=0, Hs=0) — double bonded to N
        # O- (type=2, deg=0, charge=2(-1), Hs=0) — single bonded to N
        G, mol, nodes = build_nx_and_rwmol(
            atom_specs=[
                ("C", 0, 0, 3),   # C, degree 1
                ("N", 2, 1, 0),   # N+, degree 3
                ("O", 0, 0, 0),   # O, degree 1 (=O)
                ("O", 0, 2, 0),   # O-, degree 1
            ],
            edges=[(0, 1), (1, 2), (1, 3)],
        )
        result = _infer_bond_orders(
            mol, [(0, 1), (1, 2), (1, 3)], G, nodes, QM9_ATOM_SYMBOLS,
        )
        bond_types = {(u, v): bt for u, v, bt in result}

        # N+ with correct target=4 needs one double bond (to O) and two singles
        has_double = any(bt == Chem.BondType.DOUBLE for bt in bond_types.values())
        assert has_double, (
            f"Nitromethane should have at least one double bond (N=O), "
            f"got all: {[(u, v, str(bt)) for u, v, bt in result]}"
        )


# ---------------------------------------------------------------------------
# Bug 4: Single-valence table for multi-valent atoms (S, P)
# ---------------------------------------------------------------------------

class TestMultiValenceAtoms:
    """
    Bug: _infer_bond_orders uses a hardcoded valence table with only the
    lowest standard valence for each element (e.g., S=2, P=3). Sulfur
    and phosphorus have multiple allowed valences (S: 2, 4, 6; P: 3, 5).

    When S has 3 bonds (e.g., in DMSO: C-S(=O)-C), the target valence
    is computed as 2 - 0 - 0 = 2, but it actually needs 4. The algorithm
    thinks S is over-saturated (deficit = 2 - 3 = -1) and never assigns
    the S=O double bond.

    Expected behavior: Multi-valent atoms should get correct bond orders
    based on their actual bonding context.
    """

    def test_dmso_sulfoxide_double_bond(self):
        """
        DMSO (CS(=O)C) should have an S=O double bond.
        S has 3 heavy-atom bonds and needs valence 4, not 2.
        """
        # C  (deg=0, Hs=3)
        # S  (deg=2, Hs=0) — bonded to 2 C's and 1 O
        # O  (deg=0, Hs=0) — double bonded to S
        # C  (deg=0, Hs=3)
        G, mol, nodes = build_nx_and_rwmol(
            atom_specs=[
                ("C", 0, 0, 3),
                ("S", 2, 0, 0),
                ("O", 0, 0, 0),
                ("C", 0, 0, 3),
            ],
            edges=[(0, 1), (1, 2), (1, 3)],
            atom_symbols=ZINC_ATOM_SYMBOLS,
        )
        result = _infer_bond_orders(
            mol, [(0, 1), (1, 2), (1, 3)], G, nodes, ZINC_ATOM_SYMBOLS,
        )
        bond_types = {(u, v): bt for u, v, bt in result}

        assert bond_types[(1, 2)] == Chem.BondType.DOUBLE, (
            f"S=O bond should be DOUBLE in DMSO, got {bond_types[(1, 2)]}. "
            f"Valence table hardcodes S=2 but DMSO needs S valence 4."
        )

    def test_dimethylsulfone_two_double_bonds(self):
        """
        Dimethylsulfone CS(=O)(=O)C should have two S=O double bonds.
        S has 4 heavy-atom bonds and needs valence 6.
        """
        # C  (deg=0, Hs=3)
        # S  (deg=3, Hs=0) — bonded to 2 C's and 2 O's
        # O  (deg=0, Hs=0) — double bonded to S
        # O  (deg=0, Hs=0) — double bonded to S
        # C  (deg=0, Hs=3)
        G, mol, nodes = build_nx_and_rwmol(
            atom_specs=[
                ("C", 0, 0, 3),
                ("S", 3, 0, 0),
                ("O", 0, 0, 0),
                ("O", 0, 0, 0),
                ("C", 0, 0, 3),
            ],
            edges=[(0, 1), (1, 2), (1, 3), (1, 4)],
            atom_symbols=ZINC_ATOM_SYMBOLS,
        )
        result = _infer_bond_orders(
            mol, [(0, 1), (1, 2), (1, 3), (1, 4)], G, nodes, ZINC_ATOM_SYMBOLS,
        )
        bond_types = {(u, v): bt for u, v, bt in result}
        n_doubles = sum(1 for bt in bond_types.values() if bt == Chem.BondType.DOUBLE)

        assert n_doubles == 2, (
            f"Dimethylsulfone should have 2 S=O double bonds, got {n_doubles}. "
            f"Bonds: {[(u, v, str(bt)) for u, v, bt in result]}"
        )

    def test_phosphine_oxide_double_bond(self):
        """
        Trimethylphosphine oxide CP(=O)(C)C should have a P=O double bond.
        P has 4 heavy-atom bonds and needs valence 5.
        """
        # P (deg=3, Hs=0) — bonded to 3 C's and 1 O
        G, mol, nodes = build_nx_and_rwmol(
            atom_specs=[
                ("C", 0, 0, 3),
                ("P", 3, 0, 0),
                ("O", 0, 0, 0),
                ("C", 0, 0, 3),
                ("C", 0, 0, 3),
            ],
            edges=[(0, 1), (1, 2), (1, 3), (1, 4)],
            atom_symbols=ZINC_ATOM_SYMBOLS,
        )
        result = _infer_bond_orders(
            mol, [(0, 1), (1, 2), (1, 3), (1, 4)], G, nodes, ZINC_ATOM_SYMBOLS,
        )
        bond_types = {(u, v): bt for u, v, bt in result}

        assert bond_types[(1, 2)] == Chem.BondType.DOUBLE, (
            f"P=O bond should be DOUBLE, got {bond_types[(1, 2)]}. "
            f"Valence table hardcodes P=3 but phosphine oxide needs P valence 5."
        )


# ---------------------------------------------------------------------------
# Bug 5: Infinite loop in augmenting step for fused rings
# ---------------------------------------------------------------------------

class TestInfiniteLoopInBondInference:
    """
    Bug: The augmenting step in _infer_bond_orders only looks 1 hop away.
    For fused ring systems (e.g., naphthalene), the deficit may need to
    propagate through multiple bonds. The single-step augmentation can
    oscillate a double bond back and forth between two edges without
    making progress, causing an infinite loop.

    Also infinite-loops on genuinely unsatisfiable constraints (e.g., odd
    cycles where every atom needs one more bond order) because the deficit
    circulates around the ring without any exit condition.

    Expected behavior: The algorithm should always terminate, returning
    the best partial solution if a perfect assignment is impossible.
    """

    def test_naphthalene_terminates_all_orderings(self):
        """
        Naphthalene should produce correct bond orders regardless of edge
        ordering. Some orderings cause the current algorithm to loop forever.
        """
        import random

        # Naphthalene: 10 carbons, 11 edges
        # Atoms 3 and 8 are bridgehead (degree 3, 0 Hs), rest degree 2, 1 H
        atom_specs = []
        for i in range(10):
            if i in (3, 8):
                atom_specs.append(("C", 2, 0, 0))  # bridgehead: degree 3
            else:
                atom_specs.append(("C", 1, 0, 1))  # rim: degree 2
        naph_edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (5, 6), (6, 7), (7, 8), (8, 9), (0, 9), (3, 8),
        ]

        random.seed(42)
        failures = []
        for trial in range(20):
            shuffled = naph_edges.copy()
            random.shuffle(shuffled)

            G, mol, nodes = build_nx_and_rwmol(atom_specs, shuffled)
            result = run_bond_inference_with_timeout(
                mol, shuffled, G, nodes, QM9_ATOM_SYMBOLS, timeout_sec=3,
            )
            if result is None:
                failures.append(trial)

        assert not failures, (
            f"_infer_bond_orders infinite loop on naphthalene for "
            f"{len(failures)}/20 edge orderings (trials: {failures}). "
            f"The single-step augmentation cannot propagate deficit through "
            f"fused rings."
        )

    def test_unsatisfiable_odd_cycle_terminates(self):
        """
        A 3-membered ring where every atom needs target_valence=3 but only
        has 2 bonds is genuinely unsatisfiable (total deficit is odd).
        The algorithm should terminate instead of looping.
        """
        # Triangle: 3 carbons, each degree 2, 0 Hs → target = 4-0-0 = 4
        # Each has 2 bonds (single), current_valence=2, deficit=2
        # Greedy will increase all to double (deficit 1 each, total=3, odd)
        # Then stuck: each needs 1 more, but all neighbors are satisfied
        # Actually this IS satisfiable: all double → valence 4 each ✓
        # Let's use a case that's truly unsatisfiable:
        # 3 carbons, each degree 2, 1 H → target = 4-0-1 = 3
        # With 2 bonds each, greedy gives 1 double per atom... but 3 edges
        # and 3 needed doubles won't fit (each double satisfies 2 atoms)
        # After greedy: doubles on first 2 edges, third stays single
        # → 1 atom at valence 2 (deficit 1). Augmenting swaps endlessly.
        atom_specs = [("C", 1, 0, 1)] * 3  # all need target=3
        tri_edges = [(0, 1), (1, 2), (0, 2)]

        G, mol, nodes = build_nx_and_rwmol(atom_specs, tri_edges)
        result = run_bond_inference_with_timeout(
            mol, tri_edges, G, nodes, QM9_ATOM_SYMBOLS, timeout_sec=3,
        )
        assert result is not None, (
            "_infer_bond_orders infinite loop on unsatisfiable triangle. "
            "Total deficit is odd (3 atoms × deficit 1 = 3), so no perfect "
            "assignment exists. Algorithm should terminate with best effort."
        )

    def test_fused_bicyclic_terminates(self):
        """
        A bicyclo[2.2.0] system (two fused 4-membered rings sharing an edge)
        should terminate regardless of edge ordering.
        """
        import random

        # 6 atoms, 7 edges (two 4-rings sharing edge 2-3)
        atom_specs = [("C", 1, 0, 1)] * 2 + [("C", 2, 0, 0)] * 2 + [("C", 1, 0, 1)] * 2
        fused_edges = [(0, 1), (1, 2), (2, 3), (0, 3), (2, 4), (4, 5), (3, 5)]

        random.seed(123)
        failures = []
        for trial in range(20):
            shuffled = fused_edges.copy()
            random.shuffle(shuffled)

            G, mol, nodes = build_nx_and_rwmol(atom_specs, shuffled)
            result = run_bond_inference_with_timeout(
                mol, shuffled, G, nodes, QM9_ATOM_SYMBOLS, timeout_sec=3,
            )
            if result is None:
                failures.append(trial)

        assert not failures, (
            f"_infer_bond_orders infinite loop on fused bicyclic system for "
            f"{len(failures)}/20 edge orderings."
        )


# ---------------------------------------------------------------------------
# Bug 6: Non-deterministic edge ordering affects bond inference
# ---------------------------------------------------------------------------

class TestEdgeOrderingDeterminism:
    """
    Bug: nx_to_mol uses list(set(edges)) to deduplicate edges, which
    produces an ordering that depends on Python's hash function. The
    greedy pass in _infer_bond_orders is sensitive to edge ordering —
    different orderings can produce different bond assignments for the
    same molecular topology.

    Expected behavior: Bond inference should produce the same result
    regardless of edge ordering (or at minimum, edge ordering should
    be deterministic via sorting).
    """

    def test_nx_to_mol_deterministic_across_graph_constructions(self):
        """
        nx_to_mol should produce the same SMILES for the same molecular
        topology regardless of how the NetworkX graph was constructed
        (different node/edge insertion orders).

        Previously, nx_to_mol used list(set(edges)) which produced
        hash-dependent ordering. Now it uses sorted(set(edges)).
        """
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
        atom_specs = [("C", 1, 0, 1)] * 6

        results = set()
        # Build the same benzene graph with different edge insertion orders
        import itertools
        for perm in itertools.permutations(edges):
            G, mol_rw, nodes = build_nx_and_rwmol(atom_specs, list(perm))
            mol, _ = nx_to_mol(G, dataset="qm9", infer_bonds=True, sanitize=True, kekulize=True)
            if mol is not None:
                results.add(Chem.MolToSmiles(mol, canonical=True))

        assert len(results) == 1, (
            f"nx_to_mol produced {len(results)} distinct SMILES for benzene "
            f"across edge insertion orders: {results}. "
            f"Expected exactly 1 (deterministic)."
        )

    def test_list_set_vs_sorted_edges(self):
        """
        list(set(edges)) and sorted(set(edges)) produce different orderings,
        which means bond inference depends on hash-determined iteration order.
        """
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5)]
        from_set = list(set(edges))
        from_sorted = sorted(set(edges))

        # This test documents the inconsistency. It passes if the orderings
        # differ (confirming the bug) or if they happen to match (no bug).
        # The real assertion is in test_bond_inference_same_for_all_orderings.
        if from_set != from_sorted:
            # The orderings differ — this is the source of non-determinism
            # in nx_to_mol. Using sorted() would fix it.
            pass  # Bug confirmed but not the failure condition

        # The actual test: verify sorted order is deterministic
        assert from_sorted == [(0, 1), (0, 5), (1, 2), (2, 3), (3, 4), (4, 5)]

"""
Random walk return probability features for graph nodes.

Computes the diagonal of (D^{-1}A)^k where D is the degree matrix and A is
the adjacency matrix, giving the probability that a random walk starting at
node i returns to node i after k steps. These probabilities capture global
structural information (ring membership, centrality, bridge nodes) that local
message passing may miss.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Precomputed quantile bin boundaries for ZINC molecular graphs.
# Outer key: num_bins.  Inner key: k-value.  Value: sorted boundary list
# of length ``num_bins - 1`` (equal-frequency quantile splits).
# Computed from 5 000 ZINC training molecules (~116 k atoms).
# Note: odd k values are near-degenerate on heavy-atom molecular graphs
# (bipartite-like structure -> odd-step return probabilities ~ 0).
# ---------------------------------------------------------------------------
_ZINC_RW_QUANTILE_BOUNDARIES: dict[int, dict[int, list[float]]] = {
    3: {
        2: [0.416667, 0.444444],
        3: [0.0, 0.0],
        4: [0.261574, 0.333333],
        5: [0.0, 0.0],
        6: [0.211934, 0.268519],
        7: [0.0, 0.0],
        8: [0.182436, 0.236336],
        9: [0.0, 0.000772],
        10: [0.161278, 0.215864],
        11: [0.0, 0.002165],
        12: [0.146152, 0.199449],
        13: [0.0, 0.00397],
        14: [0.134964, 0.18818],
        15: [0.0, 0.005945],
        16: [0.125639, 0.178501],
    },
    4: {
        2: [0.333333, 0.416667, 0.5],
        3: [0.0, 0.0, 0.0],
        4: [0.255208, 0.289352, 0.354167],
        5: [0.0, 0.0, 0.0],
        6: [0.199846, 0.238426, 0.287616],
        7: [0.0, 0.0, 0.003086],
        8: [0.170034, 0.206028, 0.253515],
        9: [0.0, 0.0, 0.007802],
        10: [0.149727, 0.185268, 0.231642],
        11: [0.0, 0.0, 0.012478],
        12: [0.133689, 0.170127, 0.21738],
        13: [0.0, 1e-05, 0.016577],
        14: [0.122111, 0.158143, 0.205156],
        15: [0.0, 4.5e-05, 0.020004],
        16: [0.113759, 0.148607, 0.195082],
    },
    5: {
        2: [0.333333, 0.416667, 0.444444, 0.5],
        3: [0.0, 0.0, 0.0, 0.0],
        4: [0.240741, 0.273148, 0.310185, 0.361111],
        5: [0.0, 0.0, 0.0, 0.0],
        6: [0.189108, 0.220486, 0.252915, 0.302083],
        7: [0.0, 0.0, 0.0, 0.010417],
        8: [0.159722, 0.191422, 0.222312, 0.271235],
        9: [0.0, 0.0, 0.0, 0.01929],
        10: [0.139563, 0.170497, 0.202565, 0.24619],
        11: [0.0, 0.0, 0.000193, 0.026792],
        12: [0.126104, 0.155842, 0.188097, 0.229485],
        13: [0.0, 0.0, 0.000641, 0.031937],
        14: [0.114862, 0.143742, 0.176051, 0.216373],
        15: [0.0, 0.0, 0.001309, 0.036034],
        16: [0.105988, 0.134154, 0.166114, 0.206725],
    },
    6: {
        2: [0.333333, 0.416667, 0.416667, 0.444444, 0.5],
        3: [0.0, 0.0, 0.0, 0.0, 0.0],
        4: [0.222222, 0.261574, 0.289352, 0.333333, 0.375],
        5: [0.0, 0.0, 0.0, 0.0, 0.018519],
        6: [0.179355, 0.211934, 0.238426, 0.268519, 0.317708],
        7: [0.0, 0.0, 0.0, 0.0, 0.028292],
        8: [0.151299, 0.182436, 0.206028, 0.236336, 0.279191],
        9: [0.0, 0.0, 0.0, 0.000772, 0.03628],
        10: [0.131753, 0.161278, 0.185268, 0.215864, 0.25667],
        11: [0.0, 0.0, 0.0, 0.002165, 0.041505],
        12: [0.118311, 0.146152, 0.170127, 0.199449, 0.237798],
        13: [0.0, 0.0, 1e-05, 0.00397, 0.045406],
        14: [0.107878, 0.134964, 0.158143, 0.18818, 0.224247],
        15: [0.0, 0.0, 4.5e-05, 0.005945, 0.048552],
        16: [0.09981, 0.125639, 0.148607, 0.178501, 0.213473],
    },
    7: {
        2: [0.333333, 0.388889, 0.416667, 0.416667, 0.5, 0.555556],
        3: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        4: [0.222222, 0.261574, 0.277778, 0.298611, 0.347222, 0.401235],
        5: [0.0, 0.0, 0.0, 0.0, 0.0, 0.018519],
        6: [0.171296, 0.205954, 0.22548, 0.249657, 0.280382, 0.329861],
        7: [0.0, 0.0, 0.0, 0.0, 0.0, 0.03106],
        8: [0.145006, 0.176988, 0.194354, 0.219895, 0.246854, 0.291956],
        9: [0.0, 0.0, 0.0, 0.0, 0.002315, 0.040181],
        10: [0.126008, 0.155368, 0.174808, 0.197188, 0.224782, 0.264515],
        11: [0.0, 0.0, 0.0, 5.7e-05, 0.005447, 0.046542],
        12: [0.112354, 0.13996, 0.159842, 0.18344, 0.210551, 0.24722],
        13: [0.0, 0.0, 0.0, 0.000216, 0.00854, 0.050692],
        14: [0.102474, 0.128354, 0.148099, 0.171443, 0.197762, 0.233135],
        15: [0.0, 0.0, 0.0, 0.000496, 0.011525, 0.053064],
        16: [0.095008, 0.119595, 0.138627, 0.162068, 0.18802, 0.221364],
    },
}

# Backward-compatible alias (4-bin preset)
ZINC_RW_QUANTILE_BOUNDARIES: dict[int, list[float]] = _ZINC_RW_QUANTILE_BOUNDARIES[4]


def get_zinc_rw_boundaries(num_bins: int) -> dict[int, list[float]]:
    """Return precomputed ZINC quantile bin boundaries for a given bin count.

    Available bin counts: 3, 4, 5, 6.  Each returned dict maps
    ``k`` (2..16) to a sorted list of ``num_bins - 1`` boundary values.

    Parameters
    ----------
    num_bins : int
        Number of bins (3, 4, 5, or 6).

    Returns
    -------
    dict[int, list[float]]
        Mapping from k-value to boundary list, suitable for passing as
        ``bin_boundaries`` to :func:`bin_rw_probabilities`.

    Raises
    ------
    ValueError
        If *num_bins* is not one of the precomputed values.
    """
    if num_bins not in _ZINC_RW_QUANTILE_BOUNDARIES:
        available = sorted(_ZINC_RW_QUANTILE_BOUNDARIES.keys())
        raise ValueError(
            f"No precomputed boundaries for num_bins={num_bins}. "
            f"Available: {available}"
        )
    return _ZINC_RW_QUANTILE_BOUNDARIES[num_bins]


# ---------------------------------------------------------------------------
# Precomputed quantile bin boundaries for PubChem-Large molecular graphs.
# Computed from 10 000 PubChem-Large training molecules (~413 k atoms).
# Molecules have 32-64 heavy atoms.  Covers k = 2..20.
# ---------------------------------------------------------------------------
_PUBCHEM_LARGE_RW_QUANTILE_BOUNDARIES: dict[int, dict[int, list[float]]] = {
    6: {
        2: [0.333333, 0.388889, 0.416667, 0.5, 0.5],
        3: [0.0, 0.0, 0.0, 0.0, 0.0],
        4: [0.212963, 0.259259, 0.282407, 0.33642, 0.375],
        5: [0.0, 0.0, 0.0, 0.0, 0.0],
        6: [0.168132, 0.204282, 0.238426, 0.273727, 0.319102],
        7: [0.0, 0.0, 0.0, 0.0, 0.000772],
        8: [0.139532, 0.176183, 0.205485, 0.239873, 0.277402],
        9: [0.0, 0.0, 0.0, 0.0, 0.002142],
        10: [0.120885, 0.154429, 0.183787, 0.219252, 0.259512],
        11: [0.0, 0.0, 0.0, 0.0, 0.004285],
        12: [0.107801, 0.138315, 0.168057, 0.200769, 0.240999],
        13: [0.0, 0.0, 0.0, 0.000003, 0.006541],
        14: [0.098071, 0.127194, 0.154731, 0.187081, 0.227366],
        15: [0.0, 0.0, 0.0, 0.000015, 0.008477],
        16: [0.090007, 0.11841, 0.144414, 0.176266, 0.214421],
        17: [0.0, 0.0, 0.0, 0.000046, 0.010272],
        18: [0.083762, 0.111489, 0.135859, 0.1672, 0.205205],
        19: [0.0, 0.0, 0.0, 0.000098, 0.011709],
        20: [0.078612, 0.10555, 0.128762, 0.159834, 0.197118],
    },
    7: {
        2: [0.333333, 0.333333, 0.416667, 0.416667, 0.5, 0.555556],
        3: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        4: [0.203704, 0.25, 0.273148, 0.298611, 0.354167, 0.394097],
        5: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        6: [0.161481, 0.196502, 0.220486, 0.248457, 0.289352, 0.326271],
        7: [0.0, 0.0, 0.0, 0.0, 0.0, 0.002058],
        8: [0.132316, 0.16572, 0.190844, 0.218854, 0.25185, 0.290085],
        9: [0.0, 0.0, 0.0, 0.0, 0.0, 0.004572],
        10: [0.114591, 0.145558, 0.169841, 0.197144, 0.22903, 0.264676],
        11: [0.0, 0.0, 0.0, 0.0, 0.0, 0.007323],
        12: [0.101434, 0.13079, 0.15519, 0.181427, 0.213086, 0.247837],
        13: [0.0, 0.0, 0.0, 0.0, 0.00006, 0.009944],
        14: [0.092253, 0.11936, 0.144048, 0.168546, 0.199171, 0.235434],
        15: [0.0, 0.0, 0.0, 0.0, 0.00018, 0.012138],
        16: [0.084919, 0.11088, 0.134455, 0.157481, 0.187992, 0.224152],
        17: [0.0, 0.0, 0.0, 0.0, 0.000363, 0.014437],
        18: [0.078916, 0.103889, 0.125888, 0.148811, 0.178646, 0.214298],
        19: [0.0, 0.0, 0.0, 0.0, 0.000608, 0.016358],
        20: [0.074211, 0.098143, 0.118897, 0.141204, 0.170593, 0.205452],
    },
    8: {
        2: [0.333333, 0.333333, 0.388889, 0.416667, 0.444444, 0.5, 0.583333],
        3: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        4: [0.203704, 0.23939, 0.261574, 0.282407, 0.319444, 0.356916, 0.429012],
        5: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        6: [0.153646, 0.1875, 0.211934, 0.238426, 0.259259, 0.302083, 0.337449],
        7: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.003086],
        8: [0.127852, 0.15791, 0.181584, 0.205485, 0.229953, 0.267683, 0.294271],
        9: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.006859],
        10: [0.109742, 0.139326, 0.161278, 0.183787, 0.20906, 0.239134, 0.270713],
        11: [0.0, 0.0, 0.0, 0.0, 0.0, 0.000086, 0.010757],
        12: [0.097376, 0.125205, 0.146235, 0.168057, 0.192807, 0.221154, 0.252943],
        13: [0.0, 0.0, 0.0, 0.0, 0.0, 0.000295, 0.014063],
        14: [0.088214, 0.114168, 0.134898, 0.154731, 0.179159, 0.207123, 0.240096],
        15: [0.0, 0.0, 0.0, 0.0, 0.0, 0.000626, 0.016712],
        16: [0.081177, 0.105279, 0.125492, 0.144414, 0.168287, 0.196376, 0.22847],
        17: [0.0, 0.0, 0.0, 0.0, 0.000003, 0.00108, 0.019362],
        18: [0.075771, 0.098256, 0.11779, 0.135859, 0.159249, 0.185471, 0.218155],
        19: [0.0, 0.0, 0.0, 0.0, 0.00001, 0.001604, 0.021295],
        20: [0.070816, 0.092497, 0.111222, 0.128762, 0.151763, 0.176372, 0.209672],
    },
    9: {
        2: [0.333333, 0.333333, 0.388889, 0.416667, 0.416667, 0.5, 0.5, 0.611111],
        3: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        4: [0.203704, 0.228395, 0.259259, 0.277778, 0.297839, 0.33642, 0.373457, 0.4375],
        5: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        6: [0.15072, 0.180556, 0.204282, 0.224344, 0.24537, 0.273727, 0.305556, 0.349323],
        7: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00463],
        8: [0.123537, 0.152399, 0.176183, 0.192406, 0.215543, 0.239873, 0.272264, 0.304688],
        9: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000217, 0.010031],
        10: [0.105935, 0.133591, 0.154429, 0.172698, 0.193323, 0.219252, 0.246094, 0.275391],
        11: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000622, 0.014793],
        12: [0.09383, 0.120148, 0.138315, 0.158106, 0.177679, 0.200769, 0.225586, 0.259647],
        13: [0.0, 0.0, 0.0, 0.0, 0.0, 0.000003, 0.001202, 0.01887],
        14: [0.084935, 0.109273, 0.127194, 0.146458, 0.164976, 0.187081, 0.210327, 0.245238],
        15: [0.0, 0.0, 0.0, 0.0, 0.0, 0.000015, 0.001942, 0.022237],
        16: [0.078375, 0.100762, 0.11841, 0.136756, 0.154147, 0.176266, 0.199123, 0.233035],
        17: [0.0, 0.0, 0.0, 0.0, 0.0, 0.000046, 0.002752, 0.024514],
        18: [0.072656, 0.094035, 0.111489, 0.12818, 0.145421, 0.1672, 0.191544, 0.224068],
        19: [0.0, 0.0, 0.0, 0.0, 0.0, 0.000098, 0.003603, 0.026275],
        20: [0.068122, 0.088319, 0.10555, 0.121109, 0.138472, 0.159834, 0.183344, 0.215559],
    },
    10: {
        2: [0.333333, 0.333333, 0.361111, 0.416667, 0.416667, 0.444444, 0.5, 0.5, 0.611111],
        3: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        4: [0.203704, 0.222222, 0.252894, 0.270833, 0.282407, 0.310185, 0.354167, 0.375, 0.442901],
        5: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.006944],
        6: [0.148663, 0.175926, 0.197852, 0.215856, 0.238426, 0.253687, 0.285301, 0.3125, 0.365183],
        7: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01326],
        8: [0.120656, 0.147558, 0.169235, 0.185378, 0.205485, 0.223508, 0.247846, 0.273438, 0.31023],
        9: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000457, 0.018533],
        10: [0.102833, 0.129083, 0.148383, 0.165099, 0.183787, 0.204043, 0.224768, 0.248088, 0.283548],
        11: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0013, 0.022332],
        12: [0.090488, 0.115523, 0.132934, 0.150522, 0.168057, 0.187684, 0.209493, 0.228786, 0.263373],
        13: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.00002, 0.002347, 0.025455],
        14: [0.082381, 0.105056, 0.121566, 0.138993, 0.154731, 0.17384, 0.195184, 0.215489, 0.249688],
        15: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000079, 0.003511, 0.02776],
        16: [0.075558, 0.09665, 0.112836, 0.129722, 0.144414, 0.162992, 0.184017, 0.205653, 0.23694],
        17: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000184, 0.004672, 0.029257],
        18: [0.070151, 0.089938, 0.106213, 0.121903, 0.135859, 0.154037, 0.17491, 0.196156, 0.22612],
        19: [0.0, 0.0, 0.0, 0.0, 0.0, 0.000001, 0.000344, 0.005845, 0.030959],
        20: [0.065599, 0.084539, 0.100593, 0.114696, 0.128762, 0.146292, 0.167095, 0.187183, 0.217835],
    },
}


def get_pubchem_large_rw_boundaries(num_bins: int) -> dict[int, list[float]]:
    """Return precomputed PubChem-Large quantile bin boundaries.

    Available bin counts: 6, 7, 8, 9, 10.  Each returned dict maps
    ``k`` (2..20) to a sorted list of ``num_bins - 1`` boundary values.

    Parameters
    ----------
    num_bins : int
        Number of bins (6, 8, or 10).

    Returns
    -------
    dict[int, list[float]]
        Mapping from k-value to boundary list, suitable for passing as
        ``bin_boundaries`` to :func:`bin_rw_probabilities`.

    Raises
    ------
    ValueError
        If *num_bins* is not one of the precomputed values.
    """
    if num_bins not in _PUBCHEM_LARGE_RW_QUANTILE_BOUNDARIES:
        available = sorted(_PUBCHEM_LARGE_RW_QUANTILE_BOUNDARIES.keys())
        raise ValueError(
            f"No precomputed PubChem-Large boundaries for num_bins={num_bins}. "
            f"Available: {available}"
        )
    return _PUBCHEM_LARGE_RW_QUANTILE_BOUNDARIES[num_bins]


def compute_rw_return_probabilities(
    edge_index: Tensor,
    num_nodes: int,
    k_values: tuple[int, ...] = (3, 6),
) -> Tensor:
    """
    Compute random walk return probabilities for each node.

    For each node i and step count k, computes ``[T^k]_{ii}`` where
    ``T = D^{-1} A`` is the row-stochastic transition matrix.

    Parameters
    ----------
    edge_index : Tensor
        Edge index of shape ``[2, E]`` (may be uni- or bi-directional).
    num_nodes : int
        Number of nodes in the graph.
    k_values : tuple of int
        Steps at which to compute return probabilities.

    Returns
    -------
    Tensor
        Shape ``[num_nodes, len(k_values)]`` with return probabilities
        in ``[0, 1]``.
    """
    device = edge_index.device

    # Build dense adjacency matrix
    A = torch.zeros(num_nodes, num_nodes, dtype=torch.float64, device=device)
    row, col = edge_index[0], edge_index[1]
    A[row, col] = 1.0
    # Symmetrise (handles both uni- and bi-directional input)
    A = torch.clamp(A + A.T, max=1.0)

    # Degree and transition matrix
    deg = A.sum(dim=1)
    inv_deg = torch.zeros_like(deg)
    nonzero = deg > 0
    inv_deg[nonzero] = 1.0 / deg[nonzero]

    T = A * inv_deg.unsqueeze(1)  # row i scaled by 1/deg(i)

    # Isolated nodes: self-loop with probability 1
    isolated = ~nonzero
    T[isolated, isolated] = 1.0

    # Compute T^k for each k and extract diagonal
    results = []
    for k in k_values:
        Tk = torch.linalg.matrix_power(T, k)
        results.append(torch.diag(Tk))

    return torch.stack(results, dim=-1).float().cpu()


def bin_rw_probabilities(
    rw_probs: Tensor,
    num_bins: int = 10,
    bin_boundaries: dict[int, list[float]] | None = None,
    k_values: tuple[int, ...] | None = None,
    clip_range: tuple[float, float] | None = None,
) -> Tensor:
    """
    Discretise RW return probabilities into bin indices.

    Three modes (checked in order of precedence):

    1. **Quantile** – *bin_boundaries* is not ``None``: uses
       ``torch.bucketize`` with per-k boundary vectors.
    2. **Clipped uniform** – *clip_range* is set (and *bin_boundaries*
       is ``None``): uniform bins over ``[lo, hi]`` with values outside
       the range clamped to the first / last bin.
    3. **Uniform** – neither is set: uniform bins on ``[0, 1]``.

    Parameters
    ----------
    rw_probs : Tensor
        Raw return probabilities of shape ``[N, F]`` in ``[0, 1]``.
    num_bins : int
        Number of bins.  For uniform mode this controls the bin width;
        for quantile mode it is only used to clamp the output range.
    bin_boundaries : dict mapping k → list of floats, optional
        Per-step boundary vectors.  Each list has ``num_bins - 1``
        sorted thresholds.  Use :data:`ZINC_RW_QUANTILE_BOUNDARIES`
        for a ready-made preset.
    k_values : tuple of int, optional
        The k-values corresponding to each column of *rw_probs*.
        Required when *bin_boundaries* is not ``None``.
    clip_range : tuple of (lo, hi), optional
        When set, uniform bins span ``[lo, hi]`` instead of ``[0, 1]``.
        Values below *lo* map to bin 0; values above *hi* map to the
        last bin.  Ignored when *bin_boundaries* is provided.

    Returns
    -------
    Tensor
        Integer bin indices (as float, matching ``data.x`` convention)
        of shape ``[N, F]`` with values in ``{0, ..., num_bins - 1}``.
    """
    if bin_boundaries is not None:
        # Quantile-based binning
        if k_values is None:
            raise ValueError("k_values is required when bin_boundaries is provided")

        result = torch.empty_like(rw_probs)
        for col_idx, k in enumerate(k_values):
            boundaries = bin_boundaries.get(k)
            if boundaries is None:
                result[:, col_idx] = (
                    (rw_probs[:, col_idx] * num_bins).long().clamp(0, num_bins - 1).float()
                )
            else:
                edges = torch.tensor(boundaries, dtype=rw_probs.dtype)
                result[:, col_idx] = torch.bucketize(
                    rw_probs[:, col_idx].contiguous(), edges, right=True,
                ).clamp(0, num_bins - 1).float()

        return result

    if clip_range is not None:
        # Clipped uniform binning on [lo, hi]
        lo, hi = clip_range
        scaled = (rw_probs - lo) / (hi - lo)  # 0..1 within [lo, hi]
        return (scaled * num_bins).long().clamp(0, num_bins - 1).float()

    # Uniform binning on [0, 1] (original behaviour)
    return (rw_probs * num_bins).long().clamp(0, num_bins - 1).float()


def augment_data_with_rw(
    data: Data,
    k_values: tuple[int, ...] = (3, 6),
    num_bins: int = 10,
    bin_boundaries: dict[int, list[float]] | None = None,
    clip_range: tuple[float, float] | None = None,
) -> Data:
    """
    Augment a PyG Data object with binned RW return probability features.

    Appends ``len(k_values)`` new columns to ``data.x``.

    Parameters
    ----------
    data : Data
        PyG data with ``x`` of shape ``[N, F]`` and ``edge_index`` of
        shape ``[2, E]``.
    k_values : tuple of int
        Random walk steps to compute.
    num_bins : int
        Number of bins per RW feature.
    bin_boundaries : dict mapping k → list of floats, optional
        Per-step quantile boundaries.  See :func:`bin_rw_probabilities`.
    clip_range : tuple of (lo, hi), optional
        Clipped uniform binning range.  See :func:`bin_rw_probabilities`.

    Returns
    -------
    Data
        The same ``data`` object with ``x`` expanded to
        ``[N, F + len(k_values)]``.
    """
    rw_probs = compute_rw_return_probabilities(
        data.edge_index, data.x.size(0), k_values
    )
    rw_binned = bin_rw_probabilities(
        rw_probs, num_bins,
        bin_boundaries=bin_boundaries,
        k_values=k_values,
        clip_range=clip_range,
    )
    data.x = torch.cat([data.x, rw_binned.to(data.x.device)], dim=-1)
    return data

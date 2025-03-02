import py_raccoon as pr
import itertools

from typing import Dict, Any
import pandas as pd
from ast import literal_eval
import networkx as nx
import numpy as np
import cell_flower as cf
from cell_flower import CellComplex
from scipy.sparse import hstack
from typing import List

# Different imports fix problems with snakemake
# vs interactive
try:
    from util import remove_two_cells
    from commons import DisconnectedGraph, GraphClass
except ImportError:
    from .util import remove_two_cells
    from .commons import DisconnectedGraph, GraphClass

"""
A collection of functions that are used to create problem instances
for the cell inference problem, i.e. CC, F
"""


def seed_to_rng(seed: np.random.Generator | int | None = None) -> np.random.Generator:
    """
    Takes a seed and returns a new numpy
    """
    if seed is None:
        return np.random.default_rng()
    elif type(seed) is np.random.Generator:
        return seed
    else:
        return np.random.default_rng(seed)


def sample_independent_cells(
    G: nx.Graph,
    amount: int,
    seed: np.random.Generator | int | None = None,
    oversample_factor: int = 4,
) -> List[tuple[int, ...]]:
    """
    py_raccoon samples random cells. Those are not necesarilly indpendently.
    This function changes that
    Parameters
    ----------
    G : nx.Graph
        The Graph to sample from

    amount : int
        The amount of cells to sample

    seed : np.random.Generator | int | None = None
        RNG for py_raccoon

    oversample_factor : int = 4
        The amount of cells to addtionally sample from py_raccoon
        to be moderately sure to have sampled enough independent cells



    Return
    ------
    List[tuple[int, ...]]
        A list of indpendent 2-cells in G, expected length is amount
    """

    def lin_independent(matrix, vector):
        """
        Test if a vector is linearly independent in the vectors of a matrix
        """
        stacked_matrix = hstack((matrix, vector)).todense()
        rank_stacked = np.linalg.matrix_rank(stacked_matrix)
        if rank_stacked > rank:
            return True
        return False

    er_n, er_p = pr.utils.estimate_er_params(G)  # type: ignore
    _, cells, _, _ = pr.uniform_cc(
        er_n,
        er_p,
        N=oversample_factor * amount,
        samples=2 * oversample_factor * amount,
        G=G,
        seed=seed,
    )
    cc, _, map_cells_to_index = cf.nx_graph_to_cc(G)
    rank = 0
    mapped_cells = [
        tuple(map_cells_to_index[cell[i]] for i in range(len(cell))) for cell in cells
    ]
    independent_cells = []
    boundary = cc.boundary_map(2)  # conviently get empty matrix of right shape
    for cell in mapped_cells:
        if rank >= amount:
            break
        new_boundary = cc.calc_cell_boundary(cell)
        if lin_independent(boundary, new_boundary):
            rank += 1
            boundary = hstack((boundary, new_boundary))
            independent_cells.append(cell)

    map_index_to_cells = {v: k for k, v in map_cells_to_index.items()}
    remapped_indep_cells = [
        tuple(map_index_to_cells[cell[i]] for i in range(len(cell)))
        for cell in independent_cells
    ]
    if len(remapped_indep_cells) < amount:
        print("[WARN]: Fewer cells sampled than given")
    return remapped_indep_cells[:amount]


def _flow_to_cc(
    CC: CellComplex,
    cells: List[tuple[int, ...]],
    flows: int,
    seed: int | np.random.Generator | None,
    flow_mult: float = 2.0,
    flow_add: float = 0.0,
    noise_mult: float = 1.0,
    independent_flows: bool = False,
    remove_unused_edges: bool = False,
) -> tuple[CellComplex, np.ndarray]:
    """
    Given a CellComplex without any 2-cells yet
    and list of 2-cells to add and an amount of flows:
    Adds the 2-cells to CC and creates a Flow Matrix
    Each Flow is initially chosen as a normal distribution
    centered at 0
    Then for each 2-cell another normal flow is added for each flow
    to every edge according to the boundary.
    Parameters
    CC : CellComplex
        The cell complex to compute flow for
    cells : List[tuple[int, ...]]
        A list of 2-cells to add additional flow to
    flows : int
        The amount of flows to determine
    seed : int | None = 0
        Seed for randomness. None is a fresh random seed
    flow_bonus : float = 1.0,
        Usually a normally distributed flow is added to the 2-cells
        This is a multiplicative factor
    f_transpose: bool = False
        If True output of shape (#edges, #flows)
        Else shape (#flows, #edges). For cell_flower

    Returns
    CellComplex
        CC with added 2-cells
    np.ndarray
        The Flow matrix
    """
    if flows == -1:
        return truly_uncorrelated_flows_to_cc(CC, cells, seed, noise=noise_mult)
    g = cf.cc_to_nx_graph(CC)
    if not nx.is_connected(g):
        raise DisconnectedGraph("Generated Graph was not connected")
    rng = seed_to_rng(seed)

    def random_factor():
        rand = rng.normal(scale=flow_mult)
        if flow_mult != 0:
            rand += np.sign(rand) * flow_add
        else:
            rand += flow_add
        return rand

    if remove_unused_edges:
        used_nodes = list(set([node for cell in cells for node in cell]))
        mapping = {used_nodes[i]: i for i in range(len(used_nodes))}
        used_CC = cf.CellComplex(len(used_nodes), cells=[])
        mapped_cells = [tuple(mapping[node] for node in cell) for cell in cells]
        for cell in mapped_cells:
            used_CC = used_CC.add_cell(cell)
        used_CC = remove_two_cells(used_CC)
        cells = mapped_cells
        g = cf.cc_to_nx_graph(used_CC)
        if not nx.is_connected(g):
            raise DisconnectedGraph("After removing unused edges, no longer connected")
        CC = used_CC

    F = rng.normal(size=(len(CC.get_cells(1)), flows), scale=noise_mult)
    for cell in cells:
        # insert cells into CC
        cell_boundary = CC.calc_cell_boundary(cell)
        CC = CC.add_cell_fast(cell, cell_boundary)
        if independent_flows is True:
            F += hstack([cell_boundary * random_factor() for _ in range(flows)])
        elif independent_flows is False:
            F += hstack([cell_boundary for _ in range(flows)]) * random_factor()
        else:
            raise ValueError("independent_flows must be a boolean")
    return CC, F


def truly_uncorrelated_flows_to_cc(
    CC: cf.CellComplex,
    cells: List[tuple[int, ...]],
    seed: np.random.Generator | int | None = None,
    noise: float = 1e-10,
) -> tuple[CellComplex, np.ndarray]:
    """
    Creates truly uncorrelated by creating a C that contains each possible
    vector of combination -1, 1.
    Multilply that with the boundary 2 of CC with 2-cells from cells
    B is shape (#edges, #2-cells) and C is shape (#2-cells, #combinations)

    """
    rng = seed_to_rng(seed)

    g = cf.cc_to_nx_graph(CC)
    if not nx.is_connected(g):
        raise DisconnectedGraph("Generated Graph was not connected")
    for cell in cells:
        CC = CC.add_cell(cell)
    B = CC.boundary_map(2).todense().astype(np.float64)  # type: ignore
    # C is a matrix of all combinations of -1, 1
    C = np.array(list(itertools.product([-1, 1], repeat=len(cells)))).T
    mat_noise = rng.normal(size=C.shape, scale=noise)
    C = C + mat_noise

    return CC, B @ C


def cc_with_flow(cc_config: Dict[Any, Any]) -> tuple[CellComplex, np.ndarray]:
    """
    maps the str identifier of generator to the actual generator
    and calls the generator with the param dict cc_config
    The order of edges/cells corresponds to the rows of the matrix

    Allowed str for generator are: er, watts, ba, taxi
    See their corresponding functions for more info

    Parameters
    ----------

    generator : str
        The str idetifier of the generator function

    cc_config : Dict[Any, Any]
        The param dict for the generator

    Returns
    -------
    CellComplex
        The CellComplex with sampled 2-cells

    np.ndarray
        The correstponding flow matrix

    Raises
    ------
    NotImplementedError
        If the str identifier of generator is not known
    """
    graph_class = cc_config["graph_class"]
    if graph_class == GraphClass.ER:
        return er_cc_with_flow(**cc_config)
    elif graph_class == GraphClass.WS:
        return watts_cc_with_flow(**cc_config)
    elif graph_class == GraphClass.BA:
        return ba_cc_with_flow(**cc_config)
    elif graph_class == GraphClass.TAXI:
        return taxi_cc(**cc_config)
    raise NotImplementedError(f"{graph_class} not known, please see commons.py")


def er_cc_with_flow(
    n: int,
    p: float,
    two_cells: int,
    flows: int,
    seed: int | np.random.Generator | None = None,
    flow_mult: float = 2.0,
    flow_add: float = 0.0,
    noise_mult: float = 1.0,
    independent_flows: bool = False,
    remove_unused_edges: bool = False,
    **_kwargs,
) -> tuple[CellComplex, np.ndarray]:
    """
    Create a CellComplex with an erdös-renyi graph as 1-skeleton

    Parameters
    ----------
    n : int
        Node Amount
    p : float
        probability of edge connection. Therefore, 0 <= p <= 1
    two_cells : int
        Amount of two_cells (estimate, could be smaller)
    flows : int
        The amount of flows to determine
        flows == -1 is a special value, pls see truly_uncorrelated_flows_to_cc
    seed : int | np.random.Generator | None
        Seed for randomness. None is a fresh random seed
    flow_mult : float
        Usually a normally distributed flow is added to the 2-cells
        This determines the variance sigma^2
    flow_add : float
        Adds a constant factor to each flow
    noise_mult: flaot
        The variance sigma^2 of a normal distributed noise on each edge
    independent_flows: bool
        If True, each flow gets its own random flow. Otherwise the flow on
        each 2-cells is the same
    remove_unused_edges: bool
        If True, only the edges that are used in the 2-cells are kept in the graph

    Returns
    -------
    CellComplex

    np.ndarray
        The Flow matrix

    Raises
    ------
    DinconnectedGraph
        if remove_unused_edges true and graph is then disconnected
    """
    assert (0 <= p) and (p <= 1), "p must be a probability"
    rng = seed_to_rng(seed)
    G = nx.erdos_renyi_graph(n, p, seed=rng)
    cells = sample_independent_cells(G, two_cells, seed=rng, oversample_factor=4)
    CC, *_ = cf.nx_graph_to_cc(G)
    return _flow_to_cc(
        CC,
        cells,
        flows,
        seed=seed,
        flow_mult=flow_mult,
        flow_add=flow_add,
        noise_mult=noise_mult,
        independent_flows=independent_flows,
        remove_unused_edges=remove_unused_edges,
    )


def ba_cc_with_flow(
    n: int,
    m: int,
    two_cells: int,
    flows: int,
    seed: int | np.random.Generator | None = None,
    flow_mult: float = 2.0,
    flow_add: float = 0.0,
    noise_mult: float = 1.0,
    independent_flows: bool = False,
    remove_unused_edges: bool = False,
    **_kwargs,
) -> tuple[CellComplex, np.ndarray]:
    """
    Create a CellComplex with an Barabsi Albert graph as 1-skeleton

    Parameters
    ----------
    n : int
        Node Amount
    m: int
        The amount of edges to attach from a new node
    two_cells : int
        Amount of two_cells (estimate, could be smaller)
    flows : int
        The amount of flows to determine
        flows == -1 is a special value, pls see truly_uncorrelated_flows_to_cc
    seed : int | np.random.Generator | None
        Seed for randomness. None is a fresh random seed
    flow_mult : float
        Usually a normally distributed flow is added to the 2-cells
        This determines the variance sigma^2
    flow_add : float
        Adds a constant factor to each flow
    noise_mult: flaot
        The variance sigma^2 of a normal distributed noise on each edge
    independent_flows: bool
        If True, each flow gets its own random flow. Otherwise the flow on
        each 2-cells is the same
    remove_unused_edges: bool
        If True, only the edges that are used in the 2-cells are kept in the graph

    Returns
    -------
    CellComplex

    np.ndarray
        The Flow matrix

    Raises
    ------
    DinconnectedGraph
        if remove_unused_edges true and graph is then disconnected
    """
    assert 1 <= m and m < n

    rng = seed_to_rng(seed)
    G = nx.barabasi_albert_graph(n, m, seed=rng)
    cells = sample_independent_cells(G, two_cells, seed=rng, oversample_factor=4)
    CC, *_ = cf.nx_graph_to_cc(G)
    return _flow_to_cc(
        CC,
        cells,
        flows,
        seed=seed,
        flow_mult=flow_mult,
        flow_add=flow_add,
        noise_mult=noise_mult,
        independent_flows=independent_flows,
        remove_unused_edges=remove_unused_edges,
    )


def watts_cc_with_flow(
    n: int,
    p: float,
    k: int,
    two_cells: int,
    flows: int,
    seed: int,
    flow_mult: float = 2.0,
    flow_add: float = 0.0,
    noise_mult: float = 1.0,
    independent_flows: bool = False,
    remove_unused_edges: bool = False,
    **_kwargs,
) -> tuple[CellComplex, np.ndarray]:
    """
    Create a CellComplex with an Barabsi Albert graph as 1-skeleton

    Parameters
    ----------
    n : int
        Node Amount
    p : foat
        Rewiring probbility
    k : int
        Each node is connected to k nearest neighbors in cycle topology
    two_cells : int
        Amount of two_cells (estimate, could be smaller)
    flows : int
        The amount of flows to determine
        flows == -1 is a special value, pls see truly_uncorrelated_flows_to_cc
    seed : int | np.random.Generator | None
        Seed for randomness. None is a fresh random seed
    flow_mult : float
        Usually a normally distributed flow is added to the 2-cells
        This determines the variance sigma^2
    flow_add : float
        Adds a constant factor to each flow
    noise_mult: flaot
        The variance sigma^2 of a normal distributed noise on each edge
    independent_flows: bool
        If True, each flow gets its own random flow. Otherwise the flow on
        each 2-cells is the same
    remove_unused_edges: bool
        If True, only the edges that are used in the 2-cells are kept in the graph

    Returns
    -------
    CellComplex

    np.ndarray
        The Flow matrix

    Raises
    ------
    DinconnectedGraph
        if remove_unused_edges true and graph is then disconnected
    """
    rng = seed_to_rng(seed)
    G = nx.connected_watts_strogatz_graph(n, k, p, seed=rng, tries=300)
    # for some reason pyright does not recognize py-raccoons utils. Hence ignore
    cells = sample_independent_cells(G, two_cells, seed=rng, oversample_factor=4)
    CC, *_ = cf.nx_graph_to_cc(G)
    return _flow_to_cc(
        CC,
        cells,
        flows,
        seed=seed,
        flow_mult=flow_mult,
        flow_add=flow_add,
        noise_mult=noise_mult,
        independent_flows=independent_flows,
        remove_unused_edges=remove_unused_edges,
    )


# All credit goes to https://github.com/josefhoppe/edge-flow-cell-complexes
def taxi_cc(
    flows: int,
    seed: int | np.random.Generator | None = None,
    **kwargs,
) -> tuple[CellComplex, np.ndarray]:
    """
    given the paths to the flows and graph
    taxi data is transformed into a CellComplex with flow
    The 1000 original flows are added to build flows amount of flows
    The amount that is combined is as equal as possible
    However the actual flows that are combined can be seeded

    Parameters
    ----------
    graph_path : str
        Path to the taxi graph data
    flows_path : str
        Path to the taxi flow data
    flows : int
        The amount of flows to determine
    seed : int | None = 0
        Seed for randomness. None is a fresh random seed

    Returns
    -------
    CellComplex

    np.ndarray
        The Flow matrix
    """
    if "input" in kwargs:
        graph_path = kwargs["input"][0]
        flows_path = kwargs["input"][1]
    else:
        graph_path = kwargs["graph_path"]
        flows_path = kwargs["flows_path"]

    rng = seed_to_rng(seed)

    with open(graph_path, "r") as gp:
        cells = literal_eval(gp.read())

    cc = cf.cell_complex.CellComplex(
        cells=cells, node_count=len([node for node in cells if len(node) == 1])
    )
    F = pd.read_csv(flows_path, index_col=0).fillna(0)
    F = F.groupby(rng.permutation(F.index) % flows).sum()
    F = F.to_numpy()
    return cc, F.T

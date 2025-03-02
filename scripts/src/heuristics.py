from typing import Literal, List, Any
from collections import defaultdict
import warnings

import networkx as nx
import numpy as np
import scipy
from cell_flower import CellComplex
from sklearn.decomposition import FastICA, SparsePCA

# Different imports fix problems with snakemake
# vs interactive
try:
    from .commons import (
        DiscretizingMethod,
        LRMethod,
        CandidateSelection,
        IncompatibleParameters,
    )
except ImportError:
    from commons import (
        DiscretizingMethod,
        LRMethod,
        CandidateSelection,
        IncompatibleParameters,
    )

"""
Heuristics for:
    * Choosing Vectors
    * discretizing continuous Vectors
    * (constrained) Matrix Approximation
"""


def _best_boundary_l1(
    M: np.ndarray,
    n: int = 1,
    axis: Literal[0] | Literal[1] = 0,
) -> np.ndarray:
    """
    Given a matrix returns indices of n vectors in direction ord
    which are the biggest according to the l1 norm
    Parameters
    M : np.ndarray
        The Matrix for which the indices should be returned
    n : int
        The amount of indices should be returned
    axis : 0 or 1
        0 for columns, 1 for rows
    Returns : np.ndarray
        An array containing the indices
    """
    norms = np.linalg.norm(M, ord=1, axis=axis)
    important_indices = np.argsort(norms)[::-1][:n]
    return important_indices


def entr(p):
    # ignore warnings as they are fixed by nan_to_num immediately
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", r"divide by zero encountered in log2")
        warnings.filterwarnings("ignore", r"invalid value encountered in multiply")
        return np.nan_to_num((-p * np.log2(p)))


def _best_boundary_close_discrete(
    M: np.ndarray,
    n: int = 1,
) -> np.ndarray:
    """
    Use the entropy of the normed (absolute) scalar to determine loss
    Hope: Values close to +/- 0.5 are punished the most
    Parameters
    ---------
    M : np.ndarray
        The Matrix for which to determine best vectors
    n : int
        The amount of indices

    Returns
    -------
    np.ndarray : An array containing the indices
    """

    # make non-negative
    M = np.abs(M)
    # normalize to values [0,1]
    M = M / M.max(axis=0)
    # elementwise entropy. Idea: Punish values close to .5
    Mc = np.copy(M)
    M = entr(M) + entr(np.ones_like(Mc) - Mc)
    M = np.sum(M, axis=0)
    important_indices = np.argsort(M)[:n]
    return important_indices


def _best_boundary_approx_f(
    B: np.ndarray, C: np.ndarray, F: np.ndarray, n: int
) -> np.ndarray:
    """
    Take the indices of the bounds/cycle flow combination
    which best approximates F
    For SVD it should always be [0..n].

    Parameters
    ---------
    B : np.ndarray
        The continuous boundary matrix
    C : np.ndarray
        The computed cycle flow to B
    F : np.ndarray
        The overall Flow
    n : int
        The amount of indices

    Returns
    -------
    np.ndarray : An array containing the indices
    """
    results = []
    m, f = F.shape
    for i, col in enumerate(B.T):
        results.append(np.sum(np.abs(F - col.reshape(m, 1) @ C[i, :].reshape(1, f))))
    return np.argsort(results)[:n]


def _best_boundary_weighted(
    B: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    n: int,
    w_approx_F: float,
    w_close_discrete: float,
    w_boundary_one: float,
    boundary_one: np.ndarray | scipy.sparse.csc_matrix,
) -> np.ndarray:
    """
    Computes the n best columns of B based on three differently weighted losses
    w_approx_F: How well the column-row prodct approximates F
    w_close_discrete: How close the values of the column are to {-1,0,1}
    w_one_boundary: How well the boundary condition (B_1 @ B = 0) is satisfied
    Unfortunately these losses are not normalized and some are inherently bigger
    than the others

    Parameters
    ----------
    B : np.ndarray
        The continuous boundary matrix

    C : np.ndarray
        The corresponding cycle flow

    F : np.ndarray
        The observed/overall/current harmonic flow

    n: int
        How many boundaries should be returned

    w_approx_F: float
    w_close_discrete: float
    w_boundary_one: float

    boundary_one : np.ndarray | csc_matrix
        The node-to-edge incidence matrix of the
        unevaluated cell complex

    Returns
    -------
    np.ndarray :
        An array containing the indices
    """
    results = []
    m, f = F.shape
    for i, col in enumerate(B.T):
        loss = 0
        if w_approx_F > 0:
            loss += w_approx_F * np.sum(
                np.abs(F - col.reshape(m, 1) @ C[i, :].reshape(1, f))
            )
        if w_boundary_one > 0:
            loss += w_boundary_one * np.sum(np.abs(boundary_one @ col))
        if w_close_discrete > 0:
            col = np.abs(col)
            # normalize to values [0,1]
            col = col / col.max()
            # elementwise entropy. Idea: Punish values close to .5
            Mc = np.copy(col)
            col = entr(col) + entr(np.ones_like(Mc) - Mc)
            loss += np.sum(col)
        results.append(loss)

    return np.argsort(results)[:n]


def best_boundary(
    B: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    n: int,
    method: CandidateSelection,
    kwds: dict[Any, Any],
    seed: int | None | np.random.Generator = None,
) -> np.ndarray:
    """
    Given the Approximated matrices F \approx B C
    returns n indices of the 'most important' vectors

    Parameters
    ---------
    B : np.ndarray
        The continuous boundary matrix
    C : np.ndarray
        The computed cycle flow to B
    F : np.ndarray
        The overall Flow
    n : int
        The amount of indices
    method: str
        BL1: Big L1 Norm in B
        CL1: Big L1 Norm in C
        BCloseDiscrete: Use element wise entropy to
            determine loss of one entry of B. Take the ones with little loss
        ApproxF: The column/row combinations which best approximate F
    kwds: dict[Any, Any]
        Additional options for some of the methods. Only applied if necessary
    Returns
    -------
    np.ndarray : An array containing the indices
    """
    if method == CandidateSelection.BL1:
        return _best_boundary_l1(B, n, 0)
    elif method == CandidateSelection.CL1:
        return _best_boundary_l1(C, n, 1)
    elif method == CandidateSelection.B_CLOSE_DISCRETE:
        return _best_boundary_close_discrete(B, n)
    elif method == CandidateSelection.APPROX_F:
        return _best_boundary_approx_f(B, C, F, n)
    elif method == CandidateSelection.WEIGHTED:
        return _best_boundary_weighted(B, C, F, n, **kwds)
    raise NotImplementedError()


def _rank_edges(vec: np.ndarray, CC: CellComplex) -> List[tuple[int, int]]:
    """
    Given a vector of a continuous boundary and a the corresponding CellComplex:
    ranks the edges according to 'importance' and returns them

    Parameters
    ---------
    vec : np.ndarray
        Shape (#edges,). Assigns each edge a utility
    CC : CellComplex
        The 1-cells/edges to be sorted

    Return
    ------
    List[tuple[int, int]] : A list containing the ranked edges
    """
    indexed_edges = [(edge, i) for i, edge in enumerate(CC.get_cells(k=1))]
    ranked_edges = sorted(indexed_edges, key=lambda x: np.abs(vec[x[1]]), reverse=True)
    return [x[0] for x in ranked_edges]


class CycleFreeGraph(Exception):
    """Small Exception for a graph with no cycle, i.e. forrest"""

    pass


def _discretize_ranked_edges(
    boundary_vec: np.ndarray, CC: CellComplex
) -> tuple[int, ...]:
    """
    Given a continuous boundary vector discretizes it
    by adding the most important edges until a (unique)
    cycle is found

    Parameter
    -------
    boundary_vec : np.ndarray
        A continuous boundary vector/ Column of B.
        Therefore of shape (#edges,)
    CC : CellComplex
        The underlying CellComplex for
    Return
    ------
    tuple[int, ...] : 2-cell
    """
    ranked_edges = _rank_edges(boundary_vec, CC)
    G = nx.Graph()
    nodes = set()

    for u, v in ranked_edges:
        if u in nodes and v in nodes:
            # The direction of the added two cell does not matter
            # The Flow just has to be negated to get the 'right' two-cell
            try:
                return tuple(nx.shortest_path(G, u, v))
            except nx.NetworkXNoPath:
                pass

        G.add_edge(u, v)
        nodes.update([u, v])

    raise CycleFreeGraph()


def _discretize_walk(
    boundary_vec: np.ndarray, CC: CellComplex, seed=None
) -> tuple[int, ...]:
    """
    Normalizes the boundary vector to [-1,1]
    Uses that as weighted probability choice for the
    next edge of a walk. Only increase confidence, if possible


    Parameter
    -------
    boundary_vec : np.ndarray
        A continuous boundary vector/ Column of B.
        Therefore of shape (#edges,)
    CC : CellComplex
        The underlying CellComplex for
    Return
    ------
    tuple[int, ...] : 2-cell
    """

    if seed is None:
        rng = np.random.default_rng()
    elif type(seed) is np.random.Generator:
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    # PERFORMANCE: do not use nx (performance)
    #
    # normalize boundary_vec to [-1, 1]
    boundary_vec = boundary_vec / np.max(np.abs(boundary_vec))

    edge_dict = defaultdict(list)
    for edge in CC.get_cells(k=1):
        edge_dict[edge[0]].append(edge)
        edge_dict[edge[1]].append((edge[1], edge[0]))

    def edges_of_node(node):
        return edge_dict[node]

    edge_to_index = {edge: index for index, edge in enumerate(CC.get_cells(k=1))}
    edge_to_index.update(
        {(edge[1], edge[0]): index for index, edge in enumerate(CC.get_cells(k=1))}
    )

    def weight_of_edge(edge) -> float:
        try:
            return boundary_vec[edge_to_index[edge]]
        except Exception as e:
            raise e

    def utility_of_edge(edge) -> float:
        """
        Returns the utility/confidence of choosing the edge (u,v)
        """
        if confidence_sum == 0 or edge[0] < edge[1]:
            return weight_of_edge(edge)
        else:
            return -weight_of_edge(edge)

    # initial edge:
    ie = CC.get_cells(k=1)[np.argmax(np.abs(boundary_vec))]
    visited_edges = [ie]
    nodes = set()
    confidence_sum = weight_of_edge(ie)

    assert isinstance(confidence_sum, float)  # to get linters right
    cycle_graph = nx.Graph()
    cycle_graph.add_edge(ie[0], ie[1])
    while True:
        # it should have found a 2-cell by now
        # unless it ran a single edge back and forth often
        # then escape using ranked_edges
        if len(visited_edges) > 2 * len(list(CC.get_cells(k=1))):
            return _discretize_ranked_edges(boundary_vec, CC)

        current_node = visited_edges[-1][-1]
        # TODO: Maybe limit potential edges to unvisited edges
        potential_edges = [
            e for e in edges_of_node(current_node) if utility_of_edge(e) > 0
        ]
        weights = [utility_of_edge(e) for e in potential_edges]
        if not potential_edges:
            potential_edges = [e for e in edges_of_node(current_node)]
            min_utility = np.min(
                [utility_of_edge(e) for e in potential_edges]
            )  # the most negative weight
            weights = [utility_of_edge(e) - min_utility for e in potential_edges]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", r"invalid value encountered in divide")
            weights = np.array(weights) / sum(weights)
        try:  # With a sparse boundary_vec the probabilities can often add up to 0
            next_edge = tuple(rng.choice(potential_edges, p=weights))
        except ValueError:
            next_edge = tuple(rng.choice(potential_edges))
        if next_edge[0] in nodes and next_edge[1] in nodes:
            try:
                cell = nx.shortest_path(cycle_graph, current_node, next_edge[1])
                if len(cell) > 2:
                    return tuple(cell)
            except nx.NetworkXNoPath:
                pass
        cycle_graph.add_edge(*next_edge)
        confidence_sum += utility_of_edge(next_edge)
        visited_edges.append(next_edge)
        nodes.update(next_edge)


def _discretize_discrete_first(
    boundary_vec: np.ndarray, CC: CellComplex, seed=None
) -> tuple[int, ...]:
    """
    The boundary vector first gest discretized by using a threshold.
    The associated edges build a directed graph. If a cycle can be found,
    return it. Otherwise, use ProbabilityWalk with the discretized vector

    Parameter
    -------
    boundary_vec : np.ndarray
        A continuous boundary vector/ Column of B.
        Therefore of shape (#edges,)
    CC : CellComplex
        The underlying CellComplex for
    Return
    ------
    tuple[int, ...] : 2-cell
    """

    def discretize_vector(x):
        max_val = np.max(np.abs(x))
        res = np.zeros_like(x)
        res[x > max_val / 2] = 1
        res[x < -max_val / 2] = -1
        return res

    g = nx.DiGraph()
    discrete_vector = discretize_vector(boundary_vec)
    for direction, edge in zip(discrete_vector, CC.get_cells(1)):
        assert edge[0] < edge[1]
        if direction == -1:
            g.add_edge(edge[1], edge[0])
        elif direction == 1:
            g.add_edge(edge[0], edge[1])
    try:
        cycle_edges = nx.find_cycle(g)
        res = tuple([ce[1] for ce in cycle_edges])
        return res
    except nx.NetworkXNoCycle:
        return _discretize_walk(discrete_vector, CC, seed=seed)


def discretize_boundary(
    boundary_vec: np.ndarray,  # shape (m, 1)
    flow_mat: np.ndarray,  # shape (1, flows)
    CC: CellComplex,
    method: DiscretizingMethod,
    kwds: dict[Any, Any],
    seed=None,
) -> tuple[int, ...]:
    """
    given a continuous boundary column, returns a (not normalized) cell
    For more info on each mode look at the respective function

    Parameter
    -------
    boundary_vec : np.ndarray
        A continuous boundary vector/ Column of B.
        Therefore of shape (#edges,)

    flow_vec : np.ndarray
        The flow vector corresponding to boundary_vec
        (row of C)
    CC : CellComplex
        The underlying CellComplex for
    mode : "RankedEdges" | "ProbabilityWalk" | "DiscreteFirst"
        "RankedEdges": See _discretize_ranked_edges
        "ProbabilityWalk": See _discretize_walk
        "DiscreteFirst": See _discretize_first

    Return
    ------
    tuple[int, ...] : 2-cell
    """
    if method == DiscretizingMethod.RANKED_EDGES:
        return _discretize_ranked_edges(boundary_vec, CC)
    elif method == DiscretizingMethod.P_WALK:
        return _discretize_walk(boundary_vec, CC, seed=seed)
    elif method == DiscretizingMethod.DISCRETE_FIRST:
        return _discretize_discrete_first(boundary_vec, CC, seed=seed)
    raise NotImplementedError()


# def lr_approximation(F: np.ndarray, method: LRMethod, kwds: dict[Any, Any]):
def lr_constrained_optimization(F: np.ndarray, method, kwds: dict[Any, Any], seed=None):
    if method == LRMethod.L1_GRAD:
        return _lr_weighted_grad(F, **kwds)
    else:
        raise IncompatibleParameters(
            "Used LRMethod which is not compatible with HarmonicMethod.OPT"
        )


def _lr_approximation_lu(F: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Approximate F with B @ C. Each rank r
    First uses SVD. Then uses LU on that boundary
    Idea: enforce sparsity

    Parameter
    --------
    F : np.ndarray
        The matrix to approximate
    r : int
        The maximum/expected rank of B and C each

    Return
    ------
    np.ndarray : B
    np.ndarray : C
    """

    B, C = _lr_approximation_svd(F, r)
    pl, u = scipy.linalg.lu(
        B,
        permute_l=True,
        overwrite_a=True,
    )
    return pl, u @ C


def _lr_weighted_grad(
    F: np.ndarray,
    r: int,
    boundary_one: scipy.sparse.csc_matrix | np.ndarray | None = None,
    fixed_B: scipy.sparse.csc_matrix | None = None,
    w_boundary_one: float = 1,
    w_approx_F: float = 1,
    w_close_discrete: float = 1,
    iterations: int = 10,
    lr: float = 1,
    decay: float = 0.9,
    verbose: int = 0,
):
    """
    Performs gradient descent for a low rank aproximation of F
    Three different losses are weighted:
    1. How well F is approximated
    2. How close the values of B are to {-1,0,1}
    3. Is the boarder condition B_1 @ B = 0 nearly satisfied (if B1 is given)
    Some vectors of the solution can be set via fixed_B
    Makes given amount of iterations. Uses the the best result of that time
    Decay factor possible
    verbose = 0: No output
    verbose >= 1: Output of loss in each iteration
    """
    m, flows = F.shape

    if fixed_B is not None:
        r_dot = fixed_B.shape[1]
    else:
        r_dot = 0

    def _close_discrete_grad(x):
        """
        Computes the gradient for values to be close to {-1,0,1}
        """
        if x < -0.5:
            return 2 * x + 2
        elif x < 0.5:
            return 2 * x
        else:
            return 2 * x - 2

    def _close_discrete_loss(x):
        """
        Computes the loss for values to be close to {-1,0,1}
        """
        if x < -0.5:
            return (x + 1) * (x + 1)
        elif x < 0.5:
            return x * x
        else:
            return (x - 1) * (x - 1)

    def close_discrete_loss_grad(B):
        grad_B = -np.vectorize(_close_discrete_grad)(B)
        loss = np.sum(np.abs(np.vectorize(_close_discrete_loss)(B)))
        return loss, grad_B

    def approx_f_loss_grad(B, C):
        diff = F - B @ C
        loss = np.sum(np.abs(diff))

        sgn = np.sign(diff)
        grad_B = -sgn @ C.T
        grad_C = -B.T @ sgn

        return loss, grad_B, grad_C

    def boundary_one_loss_grad(B):
        if boundary_one is False or boundary_one is None:
            return 0, np.zeros_like(B)

        product = boundary_one @ B
        loss = np.sum(np.abs(product))
        sgn = np.sign(product)
        grad_B = boundary_one.T @ sgn
        return loss, grad_B

    def reshape(X, filter_out=True):
        # offsets filters out the fixed_B and 'fixed_C'
        if filter_out:
            return (
                (X[m * r_dot : m * r_dot + m * r]).reshape((m, r)),
                (X[m * (r + r_dot) + r_dot * flows :]).reshape((r, flows)),
            )
        else:
            return (
                (X[: m * (r_dot + r)]).reshape((m, r + r_dot)),
                (X[m * (r_dot + r) :]).reshape((r + r_dot, flows)),
            )

    def combined_loss_grad(X):
        B, C = reshape(X, filter_out=False)
        loss, grad_B, grad_C = 0, np.zeros_like(B), np.zeros_like(C)
        if verbose >= 1:
            print(f"Size B: {np.sum(np.abs(B))}")

        if w_approx_F > 0:
            loss_f, grad_B_f, grad_C_f = approx_f_loss_grad(B, C)
            grad_B += w_approx_F * grad_B_f / np.sum(np.abs(grad_B_f))
            grad_C += w_approx_F * grad_C_f / np.sum(np.abs(grad_C_f))
            loss += w_approx_F * loss_f / loss_svd_f
            if verbose >= 1:
                print(f"Loss F: {loss_f}")
        if w_close_discrete > 0:
            loss_d, grad_B_d = close_discrete_loss_grad(B)
            # There is a signn error as evidenced by experiments, fixed with - intsead of +
            grad_B -= w_close_discrete * grad_B_d / np.sum(np.abs(grad_B_d))
            loss += w_close_discrete * loss_d / loss_svd_d
            if verbose >= 1:
                print(f"Loss Discrete: {loss_d}")
        if (
            w_boundary_one > 0
            and boundary_one is not False
            and boundary_one is not None
        ):
            loss_b, grad_B_b = boundary_one_loss_grad(B)
            grad_B += w_boundary_one * grad_B_b / np.sum(np.abs(grad_B_b))
            loss += w_boundary_one * loss_b / loss_svd_b
            if verbose >= 1:
                print(f"Loss Boundary: {loss_b}")

        grad = np.concatenate((grad_B.flatten(), grad_C.flatten()))
        return loss, grad

    B, C = _lr_approximation_svd(F, r)  # good initial value
    loss_svd_f, _, _ = approx_f_loss_grad(B, C)
    loss_svd_d, _ = close_discrete_loss_grad(B)
    loss_svd_b, _ = boundary_one_loss_grad(B)

    if fixed_B is not None:
        mask = np.concatenate(
            (
                np.zeros(m * r_dot),
                np.ones(m * r),
                np.ones(r_dot * flows),
                np.ones(r * flows),
            )
        )
        fixed_B = np.array(fixed_B)
        initial_C = np.linalg.lstsq(fixed_B, F - B @ C, rcond=None)[0]
        B = np.concatenate((fixed_B, B), axis=1)
        C = np.concatenate((initial_C, C), axis=0)
        # compensate for fixed_B in reshape
    else:
        mask = 1
    x0 = np.concatenate((B.flatten(), C.flatten()))
    result = gradient_descent(
        x0, combined_loss_grad, iterations, mask, lr=lr, decay=decay
    )

    return reshape(result, filter_out=True)


def gradient_descent(x0, loss_grad, iterations, mask, lr=1.0, decay=0.9):
    candidate = x0
    best_loss, _ = loss_grad(x0)
    best_candidate = candidate

    for _ in range(iterations):
        loss, grad = loss_grad(candidate)

        if loss < best_loss:
            best_loss = loss
            best_candidate = candidate
        else:
            lr *= decay

        candidate -= mask * grad * lr

    return best_candidate


def _lr_approximation_svd(F: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the singular value discomposition of F = U @ S @ Vt
    and returns B = U @ sqrt(S), C = sqrt(S) @ Vt
    truncated to rank r

    Parameter
    ---------
    F : np.ndarray
        The matrix to approximate

    r: int
        The desired low rank of B and C

    Return
    ------
    np.ndarray
        B
    np.ndarray
        C
    """
    U, S, Vt = np.linalg.svd(F, full_matrices=False)
    B = U[:, :r] @ np.diag(np.sqrt(S[:r]))
    C = np.diag(np.sqrt(S[:r])) @ Vt[:r, :]
    return B, C


def _lr_approximation_ica(F: np.ndarray, r: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the singular value discomposition of F = U @ S @ Vt
    and returns B = U @ sqrt(S), C = sqrt(S) @ Vt
    truncated to rank r

    Parameter
    ---------
    F : np.ndarray
        The matrix to approximate

    r: int
        The desired low rank of B and C

    Return
    ------
    np.ndarray
        B
    np.ndarray
        C
    """
    transformer = FastICA(n_components=r)
    B = transformer.fit_transform(F)
    C = transformer.mixing_.T
    return B, C


def _lr_approximation_spca(
    F: np.ndarray, r: int, alpha: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the singular value discomposition of F = U @ S @ Vt
    and returns B = U @ sqrt(S), C = sqrt(S) @ Vt
    truncated to rank r

    Parameter
    ---------
    F : np.ndarray
        The matrix to approximate

    r: int
        The desired low rank of B and C
    alpha: float
        The higher the value the sparser the outcome.
        See sklearn.decomposition.SparsePCA for more info

    Return
    ------
    np.ndarray
        B
    np.ndarray
        C
    """
    transformer = SparsePCA(n_components=r, alpha=alpha)
    B = transformer.fit_transform(F)
    C = transformer.components_
    return B, C


def lr_approximation(F: np.ndarray, method: LRMethod, kwds: dict[Any, Any], seed=None):
    """
    Approximate F with B @ C. Each rank r
    Parameter
    --------
    F : np.ndarray
        The matrix to approximate
    r : int
        The maximum/expected rank of B and C each

    mode : str
        "SVD": Singular Value Decomposition
        "Loss": As optimization problem. See _lr_approximation_loss
        "LU": svd, lu combination. See _lr_approximation_lu

    Return
    ------
    np.ndarray : B
    np.ndarray : C
    """
    if method == LRMethod.SVD:
        return _lr_approximation_svd(F, **kwds)
    elif method == LRMethod.LU:
        return _lr_approximation_lu(F, **kwds)
    elif method == LRMethod.L1_GRAD:
        return _lr_weighted_grad(F, **kwds)
    elif method == LRMethod.ICA:
        return _lr_approximation_ica(F, **kwds)
    elif method == LRMethod.SPCA:
        return _lr_approximation_spca(F, **kwds)
    raise NotImplementedError()

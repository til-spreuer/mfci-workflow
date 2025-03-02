import heapq
from copy import deepcopy
from typing import Any
import util

import cell_flower as cf
from cell_flower import CellComplex
import numpy as np
import scipy
from scipy.linalg import pinv


# Different imports fix problems with snakemake
# vs interactive
try:
    from heuristics import (
        best_boundary,
        lr_constrained_optimization,
        discretize_boundary,
        lr_approximation,
    )
    from util import harmonic_error, subspaces, remove_two_cells
    from commons import (
        HarmonicMethod,
        CandidateSelection,
        DiscretizingMethod,
        LRMethod,
        IncompatibleParameters,
    )
except ImportError:
    from .heuristics import (
        best_boundary,
        lr_constrained_optimization,
        discretize_boundary,
        lr_approximation,
    )
    from .util import harmonic_error, subspaces, remove_two_cells
    from .commons import (
        HarmonicMethod,
        CandidateSelection,
        DiscretizingMethod,
        LRMethod,
        IncompatibleParameters,
    )

"""
the matrix factorization cell inference, the same as in scripts/experiment,
but without the evaluation code
"""


def none_inference(CC: CellComplex, F: np.ndarray, *_, **__) -> CellComplex:
    """
    Does not add any 2-cell. Just returns original CC
    Only used for comparison
    """
    return CC


def _get_best_cells(
    cell_complex, flows, candidate_cells, n_best, gradient, error_before
):
    """
    Returns greedily the n_best cells

    Implicitly assumes all candidate cells are already normalized
    (otherwise None)
    """
    assert candidate_cells
    heap_cells = []
    for _ in range(n_best):
        heapq.heappush(
            heap_cells,
            (
                -error_before
                + np.spacing(
                    error_before
                ),  # heapq is changes also if equal, so use epsilon to ensure real better values
                (),
                (),
            ),
        )

    for cell in candidate_cells:
        if cell not in [c[1] for c in heap_cells]:
            cell_boundary = cell_complex.calc_cell_boundary(cell)
            error = util.harmonic_error(
                cell_complex, flows, gradient=gradient, added_boundary=cell_boundary
            )
            heapq.heappushpop(heap_cells, (-error, cell, cell_boundary))
    return heap_cells


def cycle_flow_matrix_factorization(
    CC: CellComplex,
    F: np.ndarray,
    n_candidates: int,
    n_best: int,
    harmonic_method_and_params: tuple[HarmonicMethod, dict[Any, Any]],
    candidate_selection_and_params: tuple[CandidateSelection, dict[Any, Any]],
    discretizing_method_and_params: tuple[DiscretizingMethod, dict[Any, Any]],
    lr_method_and_params: tuple[LRMethod, dict[Any, Any]],
    n: int | None = None,
    skip_eval: bool = False,
    epsilon: float | None = None,
    seed: int | np.random.Generator | None = None,
) -> CellComplex:
    """
    Returns 2-cells that approximate F on CC
    Return modified copy of CC after adding n 2-cells
    F gets low rank approximation B@C. Boundary vectors from B are chosen
    and discretized.

    For more info on the Methods look into commons.py

    Parameters:

    CC : CellComplex
        The CellComplex to add 2-cells to. Will not get modified
    F : np.ndarray
        An Array of shape (m,f) where m is the amount of edges/1-cells
        in CC and f is the amount of flows.
        Note: This is different from the convention of cell_flower
    n : int
        The Amount of 2-cells to add to CC.
        Currently the only termination condition
    r : int
        The rank of the Low Rank Approximation Matrices
    n_candidates : int
        The amount of two cells to test in each iteration
    n_best : int
        The n_best best candidates are added each iteration
    harmonic_method: HarmonicMethod
        How the harmonic flow is computed after each iteration
        None is equivalent explicit
    candidate_selection :  CandidateSelection
        How the vectors from the Boundary vector should be chosen
    discretizing_mode: DiscretizingMethod
        How a continuous vector is turned into a valid boundary of a 2-cell in CC
    lr_mode : LRMethod
        The method of the low rank approximation:

    epsilon : float | None
        Another termniation condition. If the harmonic error is smaller than epsilon

    Raises:
    IncompatibleParameters:
        If n_best > n_candidates, r >= amount of flows F.shape[1] or r < n_candidates
    """

    # Parameter error handling an warnings
    r = lr_method_and_params[1]["r"]
    if n_best > n_candidates:
        raise IncompatibleParameters(f"n_best ({n_best}) > best_of ({n_candidates})")
    if r >= F.shape[1]:
        raise IncompatibleParameters(f"r ({r}) >= flows ({F.shape[1]})")
    if r < n_candidates:
        raise IncompatibleParameters(f"r={r} < {n_candidates}=best_of")

    if harmonic_method_and_params[0] == HarmonicMethod.OPT and not (
        lr_method_and_params[0] == LRMethod.OPT
        or lr_method_and_params[0] == LRMethod.L1_GRAD
    ):
        raise IncompatibleParameters(
            f"If harmonic_method is OPT, then lr_method has to support fixed_B, {lr_method_and_params[0]} does not"
        )
    if n is None and epsilon is None:
        print("<Warn>: No explicit termination condition specified")

    if seed is None:
        rng = np.random.default_rng()
    elif type(seed) is np.random.Generator:
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    if (
        lr_method_and_params[0] == LRMethod.L1_GRAD
        or lr_method_and_params[0] == LRMethod.OPT
    ):
        # is True, in case the boundary was already given
        if (
            "boundary_one" in lr_method_and_params[1]
            and lr_method_and_params[1]["boundary_one"] is True
        ):
            lr_method_and_params = deepcopy(lr_method_and_params)
            lr_method_and_params[1]["boundary_one"] = CC.boundary_map(1)

    gradient, _, harmonic = subspaces(CC, F)
    current_cell_complex = CC
    cells_added = 0

    while n is None or cells_added < n:
        if harmonic_method_and_params[0] != HarmonicMethod.OPT:
            B, C = lr_approximation(harmonic, *lr_method_and_params, seed=rng)
        else:
            # boundary_one is already set, however the fixed_B changes
            # each iteration and has to be set here
            if (
                "fixed_B" in harmonic_method_and_params[1]
                and harmonic_method_and_params[1]["fixed_B"] is True
            ):
                tmp = lr_method_and_params[1].copy()
                tmp["fixed_B"] = current_cell_complex.boundary_map(2)
                lr_method_and_params = (lr_method_and_params[0], tmp)

            B, C = lr_constrained_optimization(F, *lr_method_and_params, seed=rng)
            C = C[(C.shape[0] - r) :, :]

        indices = best_boundary(
            B, C, F, n_candidates, *candidate_selection_and_params, seed=rng
        )

        cell_candidates = tuple(
            (
                discretize_boundary(
                    B[:, i],  # .reshape(F.shape[0], 1),
                    C[i, :],  # .reshape(1, F.shape[1]),
                    current_cell_complex,
                    *discretizing_method_and_params,
                    seed=rng,
                )
                for i in indices
            )
        )
        if skip_eval:
            cells = cell_candidates[:n_best]
        else:
            cell_candidates = [util.normalize_cell(cell) for cell in cell_candidates]

            error_before = util.harmonic_error(
                current_cell_complex, F, gradient=gradient
            )

            heap_cells = _get_best_cells(
                current_cell_complex,
                F,
                cell_candidates,
                n_best,
                gradient,
                error_before,
            )

            cells = []
            while heap_cells and len(cells) < n_best:
                _, cell, cell_boundary = heapq.heappop(heap_cells)
                if cell:
                    cells.append(cell)

        iteration_boundary = scipy.sparse.lil_matrix((F.shape[0], 0))  # only for pinv
        for cell in cells:
            if not cell:
                continue
            cell_boundary = current_cell_complex.calc_cell_boundary(cell)
            iteration_boundary = scipy.sparse.hstack(
                (iteration_boundary, cell_boundary)
            )
            current_cell_complex = current_cell_complex.add_cell_fast(
                cell, cell_boundary
            )
            cells_added += 1

        if harmonic_method_and_params[0] == HarmonicMethod.OPT:
            continue
        # calculate harmonic flow for next iteration
        if harmonic_method_and_params[0] == HarmonicMethod.PINV:
            # unfortunately no sparse inverse as they are likely dense anyway
            discrete_flow = pinv(iteration_boundary.todense()) @ B @ C
            harmonic -= iteration_boundary @ discrete_flow
        elif harmonic_method_and_params[0] == HarmonicMethod.EXPLICIT:
            harmonic = subspaces(current_cell_complex, F, gradient=gradient)[2]
    return current_cell_complex


if __name__ == "__main__":
    # A quick sanity test if excecuted and not used as lib
    import graph_generator
    from icecream import ic

    inference_config = {
        "n": 5,
        "n_best": 1,
        "n_candidates": 1,
        "harmonic_method_and_params": (HarmonicMethod.EXPLICIT, {}),
        "candidate_selection_and_params": (CandidateSelection.APPROX_F, {}),
        "discretizing_method_and_params": (DiscretizingMethod.RANKED_EDGES, {}),
        "lr_method_and_params": (LRMethod.SVD, {"r": 2}),
    }
    CC, F = graph_generator.er_cc_with_flow(n=10, p=0.5, two_cells=5, flows=3, seed=78)
    a = len(CC.get_cells(2))
    ic(harmonic_error(CC, F), a)
    tcc = remove_two_cells(CC)
    res = cycle_flow_matrix_factorization(
        tcc,
        F,
        **inference_config,
    )
    sim = cf.cell_inference_approximation(tcc, F.T, n_candidates=10, n=5)
    ic(harmonic_error(res, F))
    ic(harmonic_error(sim, F))
    ic(harmonic_error(tcc, F))

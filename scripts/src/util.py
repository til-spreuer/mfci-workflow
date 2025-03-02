import cell_flower as cf
from pathlib import Path
import os
from typing import List
from cell_flower.detection import project_flow
import scipy
import numpy as np
import dill
from scipy.sparse import csc_array

"""
The functionality of the following methods is expected to stay the same
This is different in heuristics
"""


def normalize_cell(cell):
    """
    Patched Version to handle () cell
    """
    if cell == ():
        return ()
    else:
        return cf.normalize_cell(cell)


def remove_two_cells(CC: cf.CellComplex):
    return cf.nx_graph_to_cc(cf.cc_to_nx_graph(CC))[0]


def write_cc_with_flow(CC: cf.CellComplex, F: np.ndarray, path: str | os.PathLike):
    directory = Path(path).parent
    directory.mkdir(parents=True, exist_ok=True)

    with open(path, "wb+") as f:
        dill.dump((CC, F), f)


def read_cc_with_flow(path: str | os.PathLike):
    with open(path, "rb") as f:
        CC, F = dill.load(f)
    return CC, F


def sort_cells(cells: List[tuple[int, ...]]):
    """
    Given a list of cells, returns a sorted list of normalized cells
    Helpful for printing
    Parameters
    cells : List[tuple[int, ...]]
        A list of cells
    """
    return list(sorted([cf.normalize_cell(cell) for cell in cells]))


def subspaces(
    CC: cf.CellComplex,
    F: np.ndarray,
    gradient: np.ndarray | None = None,
    curl: np.ndarray | None = None,
    harmonic: np.ndarray | None = None,
    added_boundary: np.ndarray | csc_array | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the subspaces of F in CC
    Optionally (some) subspaces can be given to ease computation
        Useful as gradient does not change by adding 2-cells to CC
    added_boundary to measure influence of not yet added cell
    if added boundary is given curl should be none, otherwise ignored

    Parameters
    CC: cf.CellComplex
        The cell complex for which to return the subspaces
    F: np.ndarray,
        The Flow space (gradient + curl + harmonic)
    gradient: np.ndarray | None = None,
        If not None: The gradient space to ease computation
    curl: np.ndarray | None = None,
        If not None: The curl space to ease computation
    harmonic: np.ndarray | None = None,
        If not None: The harmonic space to ease computation
    added_boundary: np.ndarray | None = None,
        If not None: Some boundary of cell(s) not in CC
    """

    if gradient is None:
        gradient = np.zeros(F.shape)
        for i in range(F.shape[1]):
            gradient.T[i] += project_flow(CC.boundary_map(1).T, F.T[[i]])
    if curl is None:
        curl = np.zeros(F.shape)
        if added_boundary is not None:
            b2 = scipy.sparse.hstack((CC.boundary_map(2), added_boundary))
        else:
            b2 = CC.boundary_map(2)

        for i in range(F.shape[1]):
            curl.T[i] += project_flow(b2, F.T[[i]])  # type: ignore
    if harmonic is None:
        harmonic = F - gradient - curl
    return gradient, curl, harmonic


def error_norm(M):
    return np.sqrt(np.sum(np.square(M)))


def harmonic_error(
    CC, F, gradient=None, curl=None, harmonic=None, added_boundary=None
) -> float:
    """
    Computes the l2 of the harmonic flow
    Optional subspaces can be given
    If added_boundary is given it computes the harmonic flow as if the according cells would be added to CC

    Parameters
    CC: cf.CellComplex
        The cell complex for which to return the subspaces
    F: np.ndarray,
        The Flow space (gradient + curl + harmonic)
    gradient: np.ndarray | None = None,
        If not None: The gradient space to ease computation
    curl: np.ndarray | None = None,
        If not None: The curl space to ease computation
    harmonic: np.ndarray | None = None,
        If not None: The harmonic space to ease computation
    added_boundary: np.ndarray | None = None,
        If not None: Some boundary of cell(s) not in CC
    """
    if harmonic is None:
        harmonic = subspaces(CC, F, gradient, curl, harmonic, added_boundary)[2]
    return error_norm(harmonic)


with open(os.path.join(Path(__file__).parent, "seed_list.txt"), "r") as f:
    seed_list = [int(line.strip()) for line in f]
assert len(set(seed_list)) == len(
    seed_list
), "seed_list.txt must not contain duplicates"


def seed_int(x: int) -> int:
    """
    Hashes an integer to an integer
    """
    try:
        return seed_list[x]
    except IndexError:
        raise ValueError(f"seed_list too short. can only seed up to {len(seed_list)-1}")

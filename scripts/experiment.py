import heapq
import random
import sys
import tracemalloc
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Sequence
from scipy.sparse import csc_array
from scipy.sparse.linalg import lsqr

import cell_flower as cf
import networkx as nx
import numpy as np
import py_raccoon as pr
import scipy
from cell_flower import CellCandidateHeuristic, CellComplex, CellSearchFlowNormalization
from cell_flower.detection import (
    cell_candidate_search_st,
    cell_candidate_search_triangles,
    project_flow,
    score_cells_multiple,
)

import src.graph_generator as gg
import src.util as util

# There is a problem with importing from snakemake and from anywhere else
# This is a workaround in case the classes should be used interactively for manual testing
from src.commons import (
    CandidateSelection,
    DiscretizingMethod,
    HarmonicMethod,
    IncompatibleParameters,
    LRMethod,
)
from src.heuristics import (
    best_boundary,
    discretize_boundary,
    lr_approximation,
    lr_constrained_optimization,
)

"""
No Script but provides the Experiment Classes
They are implemenatations of each method to solve the cell inference problem
They comupte the needed time, error, etc. and should provide sanity checks
for the attribute configuration and renaming parameters for convinience.
For general use: See the absract class Experiment

"""


def representable_vecs(
    truth: np.ndarray | csc_array, candidates: np.ndarray | csc_array
) -> int:
    """
    How many columns of truth can be built using vectors from candidates,
    i.e. how many columns of truth are a linear combination of candidates

    Parameters
    ----------
    truth : np.ndarray
        Two dimensional array of column vectors that should be reconstructed

    candidates : np.narray
        Two dimensional array


    Returns
    ----
    int
        count of representable vectors
    """
    count = 0
    for vec in truth.T:
        # PERFORMANCE: Should not convert to dense
        vec = scipy.sparse.csr_array(vec).reshape(-1).todense()
        r1norm = lsqr(candidates, vec)[4]
        if r1norm < 1e-4:
            count += 1
    return count


class Experiment(ABC):
    """
    Abstract class for the cell inference problem
    Takes options on how to create the cc and has to do so itself
    Runs the correstponding algorithm and saves the results
    Makes some postprocessing steps, like calculating cell lengths
    """

    _max_iter: int
    _initial_cell_complex: CellComplex
    _current_cell_complex: CellComplex
    _added_cells: list[tuple[int, ...]] | None
    _times: list[float] | None
    _flows: np.ndarray
    _errors: list[float] | None
    _init_time: float
    _trace_memory: bool
    # The lists _addded_cells,_times,_errors
    # Represent their corresponding value in each timestep up to _max_iter

    def __init__(
        self,
        cc_config: dict[str, Any],
        max_iter: int,
        seed: None | np.random.Generator | int = 0,
        trace_memory: bool = False,
    ) -> None:
        """
        A concrete class can have more parameters

        cc_config : dict[str, Any]
            the configuration/param dict for the graph generation
        max_iter : int
            The amount of cells to add
        seed : int | np.random.
            Randomn number generator or seed for the experiment
            cc_config has its own seed if wanted
        trace_memory : bool
            If the memory use shuld be estimated. Takes much longer
        """
        cc, flows = gg.cc_with_flow(cc_config)
        self._solution_cc = cc
        cc = util.remove_two_cells(cc)
        self._initial_cell_complex = cc
        self._current_cell_complex = cc
        self._flows = flows
        self._max_iter = max_iter
        self._errors = None
        self._added_cells = None
        self._cell_lengths = None
        self._times = None
        self._init_time = 0
        self._memory = None
        self._trace_memory = trace_memory
        self._representable_cells = None
        if seed is None:
            self._rng = np.random.default_rng()
        elif type(seed) is np.random.Generator:
            self._rng = seed
        else:
            self._rng = np.random.default_rng(seed)
        if self._trace_memory:
            tracemalloc.start()

    def get_errors(self) -> list[float]:
        """
        If necessary, makes a run of the experiment
        computes the errors from the added cells,
        saves and returns them
        """
        if self._times is None or self._added_cells is None:
            self.run()
        assert self._times is not None
        assert self._added_cells is not None
        harmonic = None

        if self._errors is None:
            temp_cc = self._initial_cell_complex
            gradient_flows, _, _ = util.subspaces(temp_cc, self._flows)
            errors = [
                util.harmonic_error(
                    self._initial_cell_complex,
                    self._flows,
                    gradient=gradient_flows,
                )
            ]
            for cell in self._added_cells:
                if cell:
                    temp_cc = temp_cc.add_cell(cell)
                    assert harmonic is None
                else:
                    assert cell == ()
                    # every following cell will empty as well
                    # At lest lsmr does not need to be recomputed
                    if harmonic is None:
                        _, _, harmonic = util.subspaces(
                            temp_cc, self._flows, gradient=gradient_flows
                        )
                errors.append(
                    util.harmonic_error(
                        temp_cc, self._flows, gradient=gradient_flows, harmonic=harmonic
                    )
                )
            self._errors = errors
        return self._errors

    def get_times(self) -> list[float]:
        """
        If necessary, runs the experiment
        Return the times as a list
        """
        if self._times is None:
            self.run()
            assert self._times is not None
        return [self._init_time] + self._times

    def get_added_cells(self) -> list[tuple[int, ...]]:
        """
        If necessary, runs the experiment
        Returns the 2-cells in order of adding them
        """
        if self._added_cells is None:
            self.run()
            assert self._added_cells is not None
        return self._added_cells

    def get_memory(self) -> list[float]:
        """
        If necessary, runs the experiment
        If trace_memory == False output a list of 0s
        otherwise the estimated memory usage of the algorithm
        """
        if self._memory is None:
            self.run()
            assert self._memory is not None
        return self._memory

    def get_representable_cells(self) -> list[int]:
        """
        If necessary runs the experiment
        Computes the representable cells of the soultion 2-cells
        from the added 2-cells after each iteration

        """
        if self._representable_cells is not None:
            return self._representable_cells
        if self._added_cells is None:
            self.run()
            assert self._added_cells is not None
        self._representable_cells = []
        truth_boundary = self._solution_cc.boundary_map(2)
        temp_cc = util.remove_two_cells(self._solution_cc)
        representable_cells = []
        for cell in self._added_cells:
            if cell == ():
                continue
            temp_cc = temp_cc.add_cell(cell)
            current_boundary = temp_cc.boundary_map(2)
            representable_cells.append(
                representable_vecs(truth_boundary, current_boundary)
            )
        try:
            representable_cells += [representable_cells[-1]] * (
                self._max_iter - len(representable_cells)
            )
        except IndexError:
            representable_cells += [0] * self._max_iter
        self._representable_cells = representable_cells
        return self._representable_cells

    def run(self) -> tuple[list[tuple[int, ...]], list[float]]:
        """
        Runs the experiment:
        Add _max_iter many 2-cells and measure the needed time for each iteration
        (and memory if trace_memory is True)
        Captures the added_cells but does not yet compute the errors
        Directly output the added_cells and times even though
        convinience functions are available
        """
        if self._added_cells is not None and self._times is not None:
            return self._added_cells, self._times
        self._added_cells = []
        self._times = []
        self._memory = []

        while len(self._added_cells) < self._max_iter:
            if self._trace_memory:
                tracemalloc.reset_peak()
            time_start = time()
            results = self.simulate_one_iteration()
            time_end = time()
            for next_cell in results:
                if len(self._added_cells) >= self._max_iter:
                    continue
                self._added_cells.append(next_cell)
                self._times.append((time_end - time_start) / len(results))
                if self._trace_memory:
                    self._memory.append(tracemalloc.get_traced_memory()[1])
                else:
                    self._memory.append(np.nan)

        self.check_cc()
        return self._added_cells, self._times

    def get_cell_lengths(self) -> list[int]:
        """
        If necessary, runs the experiment
        Returns the overall length cells
        So at position i of returned list is the count of how many
        2-cells are of length exactly i.
        """
        if self._cell_lengths is not None:
            return self._cell_lengths
        if self._added_cells is None:
            self.run()
        assert self._added_cells is not None
        lengths = [len(cell) for cell in self._added_cells]
        self._cell_lengths = []
        for i in range(max(lengths) + 1):
            self._cell_lengths.append(lengths.count(i))
        return self._cell_lengths

    @staticmethod
    def sanity_check(cc_config, exp_config) -> bool:
        """
        Given the paramdict of the constructed cc
        and the parameters of the algorithm to excecute can outright
        reject some non-sensical combinations,
        e.g. the cc only has two flows but the requested low rank
        of the matrix factorization approach is three.
        """
        return True

    @staticmethod
    def config_to_file_rename(config):
        """
        Allows for renaming some parameters to be
        easier humanly readable, e.g. use the names of enums not
        their import path
        """
        return config

    @staticmethod
    def file_to_config_rename(config):
        """
        Reverse to the original parameter values,
        e.g. Enum Name to actual Enum value
        """
        return config

    def check_cc(self):
        """
        A small check that is excuted after a run to see
        that the original cc was not modified,
        e.g. no accidental adding of edges
        """
        old_underlying_graph = cf.cc_to_nx_graph(self._initial_cell_complex)
        new_underlying_graph = cf.cc_to_nx_graph(self._current_cell_complex)
        if old_underlying_graph != new_underlying_graph:
            assert nx.is_isomorphic(
                old_underlying_graph, new_underlying_graph
            ), "Underlying graph should not change"

    @abstractmethod
    def simulate_one_iteration(self) -> Sequence[tuple[int, ...]]:
        """
        Simulate the iteration of the run
        Return the 2-cells added in that iteration,
        usually one but multiple possible
        """
        pass


def approach_to_experiment(approach: str) -> type[Experiment]:
    """
    Given the readable str identifier returns the actual experiment class

    Parameters
    ----------
    approach : str
        The str identifier of the experiment

    Returns
    -------
    type[Experiment]
        The experiment

    Raises
    ------
    NotImplementedError
        If the approach is not known
    """
    if approach == "mat_fact":
        return MatFactExperiment
    elif approach == "cell_flower":
        return CellFlowerExperiment
    elif approach == "none_inference":
        return NoneExperiment
    elif approach == "construction":
        return ConstructionExperiment
    elif approach == "random_inference":
        return RandomExperiment
    elif approach == "svd":
        return SVDExperiment
    else:
        raise NotImplementedError(
            f"Add your function name/call above. {approach} is unknown."
        )


class SVDExperiment(Experiment):
    _max_iter: int
    _initial_cell_complex: CellComplex
    _current_cell_complex: CellComplex
    _added_cells: list[tuple[int, ...]] | None
    _times: list[float] | None
    _flows: np.ndarray
    _errors: list[float] | None
    _init_time: float
    _trace_memory: bool

    def __init__(
        self,
        cc_config,
        max_iter: int,
        seed: int = 0,
        trace_memory: bool = False,
    ) -> None:
        super().__init__(cc_config, max_iter, seed, trace_memory)

    def get_errors(self) -> list[float]:
        if self._errors is not None:
            return self._errors
        harmonic = util.subspaces(self._current_cell_complex, self._flows)[2]
        self._errors = [util.error_norm(harmonic)]
        B, C = lr_approximation(
            harmonic,
            LRMethod.SVD,
            kwds={"r": min(*harmonic.shape, self._max_iter)},
        )

        for i in range(self._max_iter):
            self._errors.append(
                util.error_norm(harmonic - B[:, : i + 1] @ C[: i + 1, :])
            )

        return self._errors

    def get_times(self) -> list[float]:
        return [0] * (self._max_iter + 1)  # +1 because of 'init_time'

    def get_added_cells(self) -> list[tuple[int, ...]]:
        return [()] * self._max_iter

    def get_memory(self) -> list[float]:
        return [0] * self._max_iter

    def run(self) -> tuple[list[tuple[int, ...]], list[float]]:
        self._added_cells = [()] * self._max_iter
        self._times = [0.0] * (self._max_iter + 1)
        return self.get_added_cells(), self.get_times()

    def get_cell_lengths(self):
        return [0] * self._max_iter

    @staticmethod
    def sanity_check(cc_config, exp_config) -> bool:
        return True

    @staticmethod
    def config_to_file_rename(config):
        return config

    @staticmethod
    def file_to_config_rename(config):
        return config

    def simulate_one_iteration(self) -> Sequence[tuple[int, ...]]:
        return [()]


class NoneExperiment(Experiment):
    """
    Always returns empy cell
    """

    def simulate_one_iteration(self) -> Sequence[tuple[int, ...]]:
        return [()]


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


class MatFactExperiment(Experiment):
    """
    Implements the matrix factorization approach for cell inference
    as described in the thesis
    The code is mirrored in ../lib/inference.py but without the evaluation code
    """

    _n_candidates: int
    _n_best: int
    # The Workings of these different methods get explained in
    # ../lib/heuristics.py and ../lib/commons.py for an overview
    _harmonic_method_and_params: tuple[HarmonicMethod, dict[Any, Any]]
    _candidate_selection_and_params: tuple[CandidateSelection, dict[Any, Any]]
    _discretizing_method_and_params: tuple[DiscretizingMethod, dict[Any, Any]]
    _lr_method_and_params: tuple[LRMethod, dict[Any, Any]]

    @staticmethod
    def file_to_config_rename(config):
        for parameter, value in config.items():
            # Any of the given parameters has a dict as value
            # with one unique entry
            if parameter == "lr_method_and_params":
                method = getattr(LRMethod, value[0])
            elif parameter == "harmonic_method_and_params":
                method = getattr(HarmonicMethod, value[0])
            elif parameter == "discretizing_method_and_params":
                method = getattr(DiscretizingMethod, value[0])
            elif parameter == "candidate_selection_and_params":
                method = getattr(CandidateSelection, value[0])
            else:
                continue
            config[parameter] = (method, value[1])
        return config

    @staticmethod
    def config_to_file_rename(config):
        for name, value in sorted(config.items()):
            if isinstance(value, dict):
                method_number, _ = list(value.items())[0]
                if name == "lr_method_and_params":
                    method_name = LRMethod(method_number).name
                elif name == "discretizing_method_and_params":
                    method_name = DiscretizingMethod(method_number).name
                elif name == "harmonic_method_and_params":
                    method_name = HarmonicMethod(method_number).name
                elif name == "candidate_selection_and_params":
                    method_name = CandidateSelection(method_number).name
                else:
                    raise NotImplementedError(
                        f"{name} has no according enum in lib/commons"
                    )
                config[name][method_name] = config[name].pop(method_number)
        return config

    @staticmethod
    def sanity_check(cc_config, exp_config) -> bool:
        if (
            exp_config["n_candidates"] > 0
            and exp_config["n_best"] > exp_config["n_candidates"]
        ):
            return False
        if exp_config["lr_method_and_params"][1]["r"] < exp_config["n_candidates"]:
            return False
        if (
            exp_config["lr_method_and_params"][1]["r"] >= cc_config["flows"]
            and cc_config["flows"] > 0
        ):
            return False
        if (
            cc_config["flows"] == -1
            and exp_config["lr_method_and_params"][1]["r"]
            >= 2 ** cc_config["two_cells"]
        ):
            return False
        if exp_config["harmonic_method_and_params"][0] == HarmonicMethod.OPT and not (
            exp_config["lr_method_and_params"][0] == LRMethod.L1_GRAD
        ):
            """
            If the harmonic flow should be computed with fixed boundaries
            (described by HarmonicMethod.OPT) the the matrix factorization
            has to be compatible with fixed boundaries
            """
            return False
        return True

    def __init__(
        self,
        cc_config,
        max_iter,
        n_candidates,
        n_best,
        harmonic_method_and_params,
        candidate_selection_and_params,
        discretizing_method_and_params,
        lr_method_and_params,
        seed,
        trace_memory,
        *,
        _="_",
    ) -> None:
        super().__init__(cc_config, max_iter, seed, trace_memory)
        init_time_start = time()

        self._n_candidates = n_candidates
        self._n_best = n_best
        self._harmonic_method_and_params = harmonic_method_and_params
        self._candidate_selection_and_params = candidate_selection_and_params
        self._discretizing_method_and_params = discretizing_method_and_params
        self._lr_method_and_params = lr_method_and_params

        r = lr_method_and_params[1]["r"]
        self.r = r
        if n_candidates > 0 and n_best > n_candidates:
            raise IncompatibleParameters(
                f"n_best ({n_best}) > best_of ({n_candidates})"
            )
        if r >= self._flows.shape[1]:
            raise IncompatibleParameters(f"r ({r}) >= flows ({self._flows.shape[1]})")
        if r < n_candidates:
            raise IncompatibleParameters(f"r={r} < {n_candidates}=best_of")

        if harmonic_method_and_params[0] == HarmonicMethod.OPT and not (
            lr_method_and_params[0] == LRMethod.L1_GRAD
        ):
            raise IncompatibleParameters(
                f"If harmonic_method is OPT, then lr_method has to support fixed_B, {lr_method_and_params[0]} does not"
            )

        if lr_method_and_params[0] == LRMethod.L1_GRAD:
            # is True, in case the boundary was already given
            if (
                "boundary_one" in lr_method_and_params[1]
                and lr_method_and_params[1]["boundary_one"] is True
            ):
                self._lr_method_and_params = deepcopy(lr_method_and_params)
                self._lr_method_and_params[1]["boundary_one"] = (
                    self._initial_cell_complex.boundary_map(1)
                )
        if candidate_selection_and_params[0] == CandidateSelection.WEIGHTED:
            self._candidate_selection_and_params[1]["boundary_one"] = (
                self._initial_cell_complex.boundary_map(1)
            )

        self._gradient, _, self._harmonic = util.subspaces(
            self._initial_cell_complex, self._flows
        )
        init_time_end = time()
        self._init_time = init_time_end - init_time_start

    def simulate_one_iteration(self) -> Sequence[tuple[int, ...]]:
        assert self._added_cells is not None
        try:
            if self._added_cells[-1] == ():
                return [()]
        except IndexError:
            pass

        assert (
            self._harmonic_method_and_params[0] != HarmonicMethod.OPT
            or self._harmonic is not None
        )

        if self._harmonic_method_and_params[0] != HarmonicMethod.OPT:
            B, C = lr_approximation(
                self._harmonic, *self._lr_method_and_params, seed=self._rng
            )
        else:
            # boundary_one is already set, however the fixed_B changes
            # each iteration and has to be set here
            if (
                "fixed_B" in self._harmonic_method_and_params[1]
                and self._harmonic_method_and_params[1]["fixed_B"] is True
            ):
                tmp = self._lr_method_and_params[1].copy()
                tmp["fixed_B"] = self._current_cell_complex.boundary_map(2)
                self._lr_method_and_params = (self._lr_method_and_params[0], tmp)

            B, C = lr_constrained_optimization(
                self._flows, *self._lr_method_and_params, seed=self._rng
            )
            C = C[(C.shape[0] - self.r) :, :]

        indices = best_boundary(
            B, C, self._flows, self._n_candidates, *self._candidate_selection_and_params
        )

        cell_candidates = tuple(
            (
                discretize_boundary(
                    B[:, i],  # .reshape(F.shape[0], 1),
                    C[i, :],  # .reshape(1, F.shape[1]),
                    self._current_cell_complex,
                    *self._discretizing_method_and_params,
                    seed=self._rng,
                )
                for i in indices
            )
        )

        if self._n_candidates <= 0:
            # It would be smarter to only discretize the self._n_best and return that
            # instead of discretizing every vector first
            cells = cell_candidates[: self._n_best]
        else:
            cell_candidates = [util.normalize_cell(cell) for cell in cell_candidates]

            error_before = util.harmonic_error(
                self._current_cell_complex, self._flows, gradient=self._gradient
            )

            heap_cells = _get_best_cells(
                self._current_cell_complex,
                self._flows,
                cell_candidates,
                self._n_best,
                self._gradient,
                error_before,
            )

            cells = []
            while heap_cells and len(cells) < self._n_best:
                _, cell, cell_boundary = heapq.heappop(heap_cells)
                if cell:
                    cells.append(cell)
            if not cells:
                return [
                    ()
                ]  # harmonic does not have to be updated as no improving cell has been found
        iteration_boundary = scipy.sparse.lil_matrix((self._flows.shape[0], 0))
        for cell in cells:
            if not cell:  # can only happen  if skip_eval
                continue
            cell_boundary = self._current_cell_complex.calc_cell_boundary(cell)
            iteration_boundary = scipy.sparse.hstack(
                (iteration_boundary, cell_boundary)
            )
            self._current_cell_complex = self._current_cell_complex.add_cell_fast(
                cell, cell_boundary
            )
        self.update_harmonic(iteration_boundary, B, C)
        return cells

    def update_harmonic(self, iteration_boundary=None, B=None, C=None):
        if self._harmonic_method_and_params[0] == HarmonicMethod.EXPLICIT:
            self._harmonic = util.subspaces(
                self._current_cell_complex, self._flows, gradient=self._gradient
            )[2]
        elif self._harmonic_method_and_params[0] == HarmonicMethod.PINV:
            assert iteration_boundary is not None
            assert B is not None
            assert C is not None
            self._harmonic -= (
                iteration_boundary
                @ scipy.linalg.pinv(iteration_boundary.todense())
                @ B
                @ C
            )
        elif self._harmonic_method_and_params[0] == HarmonicMethod.OPT:
            pass
            # No update needed as it is part of the B, C Approximation
        else:
            raise NotImplementedError


class ConstructionExperiment(Experiment):
    """
    Returns the generating cells (or ground-truth)
    Used solely for comparison
    """

    def __init__(
        self,
        cc_config,
        max_iter,
        seed: int = 0,
        trace_memory: bool = False,
        *,
        _="_",
    ) -> None:
        super().__init__(cc_config, max_iter, seed, trace_memory)
        self._iterations = 0
        self._sorted_cells = sorted(
            self._solution_cc.get_cells(2), key=lambda x: len(x), reverse=True
        )

    def simulate_one_iteration(self) -> Sequence[tuple[int, ...]]:
        try:
            next_cell = self._sorted_cells[self._iterations]
            self._iterations += 1
            return [next_cell]
        except IndexError:
            return [()]


class RandomExperiment(Experiment):
    """
    Uses py_raccoon to sample (independent) cells
    """

    def __init__(
        self,
        cc_config,
        max_iter,
        seed: int = 0,
        trace_memory: bool = False,
        *,
        _="_",
    ) -> None:
        super().__init__(cc_config, max_iter, seed, trace_memory)
        time_start = time()
        self._iteration = 0

        G = cf.cc_to_nx_graph(self._current_cell_complex)
        from src.graph_generator import sample_independent_cells

        self._cells = sample_independent_cells(G, max_iter, self._rng)
        time_end = time()
        self._init_time = time_end - time_start

    def simulate_one_iteration(self) -> Sequence[tuple[int, ...]]:
        try:
            next_cell = self._cells[self._iteration]
        except IndexError:
            return [()]
        self._iteration += 1
        return [next_cell]


class CellFlowerExperiment(Experiment):
    """
    A copy of the code from cell_flower adapted to our needs
    """

    _n_candidates: int
    _n_clusters: int
    _heuristic: CellCandidateHeuristic
    _rng: np.random.Generator
    _no_gradient_flows: np.ndarray
    _harmonic_flows: np.ndarray
    _flow_norm: CellSearchFlowNormalization
    _epsilon: float

    @staticmethod
    def file_to_config_rename(config):
        config["heuristic"] = getattr(cf.CellCandidateHeuristic, config["heuristic"])
        return config

    @staticmethod
    def config_to_file_rename(config):
        for name, value in sorted(config.items()):
            if name == "heuristic":
                if not isinstance(value, str):  # especially for compare.py
                    config[name] = cf.CellCandidateHeuristic(value).name
        return config

    @staticmethod
    def sanity_check(cc_config, exp_config) -> bool:
        if exp_config["heuristic"] == CellCandidateHeuristic.MAX:
            if exp_config["n_clusters"] != 0:
                return False
        elif exp_config["heuristic"] == CellCandidateHeuristic.SIMILARITY:
            if exp_config["n_clusters"] == 0:
                return False
        return True

    def __init__(
        self,
        cc_config,
        max_iter,
        n_candidates,
        heuristic,
        n_clusters,
        seed,
        trace_memory,
        flow_norm=CellSearchFlowNormalization.LEN,
        *,
        _="_",
    ) -> None:
        super().__init__(cc_config, max_iter, seed, trace_memory)
        init_time_start = time()
        self._n_candidates = n_candidates
        self._n_clusters = n_clusters
        self._heuristic = heuristic
        self._flow_norm = flow_norm
        self.__cells_added = 0
        self._no_gradient_flows = np.copy(self._flows.T)
        for i in range(self._flows.T.shape[0]):
            self._no_gradient_flows[i] -= project_flow(
                self._current_cell_complex.boundary_map(1).T, self._flows.T[[i]]
            )
        self._harmonic_flows = np.copy(self._no_gradient_flows)
        for j in range(self._harmonic_flows.shape[0]):
            self._harmonic_flows[j] -= project_flow(
                self._current_cell_complex.boundary_map(2).T.T,
                self._no_gradient_flows[[j]],
            )
        init_time_end = time()
        self._init_time = init_time_end - init_time_start

    def simulate_one_iteration(self) -> Sequence[tuple[int, ...]]:
        assert self._added_cells is not None
        try:
            if self._added_cells[-1] == ():
                return [()]
        except IndexError:
            pass
        if self._heuristic == CellCandidateHeuristic.TRIANGLES:
            candidate_cells = cell_candidate_search_triangles(
                self._current_cell_complex, self._n_candidates, self._harmonic_flows
            )
        else:
            candidate_cells = cell_candidate_search_st(
                self._rng,
                self._current_cell_complex,
                self._n_candidates,
                self._harmonic_flows,
                self._heuristic,
                self._n_clusters,
                self._flow_norm,
            )
        if self._n_candidates == 1:
            next_cell = candidate_cells[0][0]
        else:
            score_vals, score_cells = score_cells_multiple(
                self._current_cell_complex,
                self._no_gradient_flows,  # type: ignore
                [cell[:2] for cell in candidate_cells],
            )
            scores = np.sum(score_vals, axis=1)
            next_cell = score_cells[np.argmin(scores)]

        cell_boundaries = {cell[0]: cell[1] for cell in candidate_cells}

        if next_cell == ():
            return [()]

        self._current_cell_complex = self._current_cell_complex.add_cell_fast(
            next_cell, cell_boundaries[next_cell]
        )

        self._harmonic_flows = np.copy(self._no_gradient_flows)
        for j in range(self._harmonic_flows.shape[0]):
            self._harmonic_flows[j] -= project_flow(
                self._current_cell_complex.boundary_map(2).T.T,
                self._no_gradient_flows[[j]],
            )
        self.__cells_added += 1
        return [next_cell]

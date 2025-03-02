from enum import IntEnum

# Use IntEnum and not Enum as this simplifies comparisons
# which otherwise are messy with different level import
# due to snakemake workflow


class GraphClass(IntEnum):
    """
    For ER, WS, BA see the corresponding generators in NetworkX:
    erdos_renyi_graph, connected_watts_strogatz_graph and barabasi_albert_graph respectively
    TAXI: combines taxi data from New-York, based on
        https://chriswhong.com/open-data/foil_nyc_taxi/

    """

    ER = 1  # Erdős–Rényi
    WS = 2  # Watts–Strogatz
    BA = 3  # Barabási-Albert
    TAXI = 4  # Taxi Graph


class HarmonicMethod(IntEnum):
    """
    How the harmonic flow is computed after each iteration
    EXPLICIT: computes the harmonic flow using the boundaries of the
                current cell complex. Is accurate
    PINV: uses the pseudo inverse of the discretized Boundary, B and C
            to approximate what the corresponding should be
    OPT: Implicti in the LR Approximation. However the LR Method has
           to support fixed_B
    default None is equivalent to "explicit"
    """

    EXPLICIT = 1
    PINV = 2
    OPT = 3


class LRMethod(IntEnum):
    """
    The method of the low rank approximation:
    SVD: Optimal r-rank approximation using the singular value decomposition
    LU: Applies LU decomposition on the SVD to enforce sparsity
    L1_GRAD: Manually computed Gradient. Gives options to weight:
            Closeness to {-1,0,1}, Approx F, B1 * B = 0
            parameters: w_approx_F, w_boundary_one, w_close_discerete,
                        iterations, lr (the learning rate), decay
    ICA: Uses sklearn.decomposition.FastICA.
         Does not support additional parameters besides r(=n_components)
    """

    SVD = 1
    LU = 2
    L1_GRAD = 3
    ICA = 4
    SPCA = 5


class CandidateSelection(IntEnum):
    """
    How the vectors from the Boundary vector should be chosen
    BL1: The L1 norm of a column is big
    CL1: The L1 norm of the corresponding flow row is big
    BCloseDiscrete: Uses elementwise entropy to determine columns
                      which mostly has values close to {-1,0,1}
    APPROX_F: Whichever column-row combination best aprroximates F
    WEIGHTED: Combines loss of APPROX_F and BCloseDiscrete and boundary condition
                Can be equivalent to APPROX_F for weights (1,0,0)
              parameters: w_approx_F, w_boundary_one and w_close_discerete
    default None is equivalent to "BL1"
    """

    BL1 = 1
    CL1 = 2
    B_CLOSE_DISCRETE = 3
    APPROX_F = 4
    WEIGHTED = 5


class DiscretizingMethod(IntEnum):
    """
    RankedEdges: Edges are added ranked according to their value in B.
                Those are added in order until the first (unique) cycle
                is found
    ProbabilityWalk: The values in B are normalized an used as
                     probability for a random walk. If the walk takes
                     too long RANKED_EDGES is used as fallback
    DiscreteFirst: The values get discretized using a threshold. This
                   forms a directed graph. If there is a directed cycle,
                   that gets turned into a 2-cell. Otherwise ProbabilityWalk
                   is used with the discretized vector
    default None is equivalent to "ProbabilityWalk"
    """

    RANKED_EDGES = 1
    P_WALK = 2
    DISCRETE_FIRST = 3


class IncompatibleParameters(Exception):
    """
    Custom Exception instead of silent error handling
    """

    pass


class DisconnectedGraph(Exception):
    """
    Exception thrown if a graph is not connected
    Raised especially for graph generation if remove_unused_edges
    results in disconnected graph
    """

    pass

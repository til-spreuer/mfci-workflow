# Matrix Factorization Cell Inference

This workflow was used to construct the figures for the paper: Faster Inference of Cell Complexes from Flows via
Matrix Factorization.

TODO: Link Paper/Preprint

The Matrix Cell Inference is a framework that infers 2-cells from a given graph with multiple edge flows $F$
by solving the continuous version of the otherwise NP-hard problem to find a valid polygon-to-edge boundary matrix $B_2$
(and the corresponding cycle flow $C$) such that $F \approx B_2 \cdot C$.
The continuous version is the known matrix factorization problem.
Then, a discretized solution (to the original problem) is extracted.
In this project we implemented this strategy as well as a workflow that compares this method with the state-of-the art algorithm
the Spanning Tree Heuristic of `cell_flower`
(see the following:
[workflow](https://github.com/josefhoppe/edge-flow-cell-complexes),
[library](https://github.com/josefhoppe/cell-flower) and
[paper](https://proceedings.mlr.press/v231/hoppe24a.html)
)

## Running the workflow

Without any configuration this workflow is targeted towards creating the plots of the accompanying paper.
However, with slight modification it allows for a variety of different comparisons.

### Local Execution

To use snakemake: <https://snakemake.readthedocs.io/en/stable/>

1. Install Miniconda/mamba or similar <https://docs.anaconda.com/miniconda/>
2. Create a conda environment and install dependencies: `conda env create -n flow -f ./environment.yaml`.
   If you have compatibility issues you might want to try `conda env create -n flow -f ./environment_explicit.yaml` instead.
3. Activate the environment: `conda activate flow`
4. Clone this repository
5. Run snakemake in the cloned repository: `snakemake all -c #CORES`

The `./configs` folder contains different combinations of graph classes and algorithms.
Every possible (and sane) combination is then computed.
A `.pdf` is created combining every such combination in a single file named `.out/plot_combined/plot_[CONFIG_NAME]`
for a configuration file named `config_[CONFIG_NAME].yaml`.
Depending on the configuration files `--allow-ambiguity` may be necessary (for step 4: Run snakemake).
More information in [Configuration](#configuring-the-workflow)

### Cluster/SLURM Execution

The slurm executor: <https://snakemake.github.io/snakemake-plugin-catalog/plugins/executor/slurm.html>
was already installed through the `enviroment.yaml`
Logged in on a login node of the slurm - cluster:
`snakemake --executor slurm raw --default-resources slurm_account=[ACCOUNT] slurm_partition=[PARTITION] tasks=1 cpus_per_task=1 --jobs 512 --latency-wait 120`
Replace [ACCOUNT] with your project id or your personal account and [PARTITION] with an allowed partition of the [ACCOUNT].

After `snakemake` created all the raw files use
`rsync -aPi -e ssh USER@DATA_CLUSTER_URL:/PATH_TO_PROJECT/out ./out/`
on your local PC from the workflow folder (or replace `./out/` with the corresponding path)

Then execute `snakemake all -c #CORES` locally
Or check that file transfer was successful with
`snakemake raw -n`
and very that no file has to be computed.
If something went wrong delete the `./.snakemake/incomplete`
directory, use the `--touch` flag or try different rerun triggers, e.g. `--rerun-triggers mtime`.
Then `./out/plot_combined/` contains the plots for every graph and algorithm execution.

### Configuring the workflow

As mentioned the `./configs` folder contains different configuration files for parameter searches.
`./configs/_documented_config.yaml` explains the different parameters and how to create/modify a set.
In the next part of this readme we present the different heuristics on a higher level.

## Heuristics

This section copies and expands on the documented configuration file.
Please note that the source code itself is also documented.

#### Graph Classes

1. `ER`: Erdős–Rényi random graphs
2. `WS`: (connected) Watts–Strogatz random graphs
3. `BA`: Barabási–Albert random graphs
4. `TAXI`: Uses the real-world data set ( <https://chriswhong.com/open-data/foil_nyc_taxi/> )

#### Harmonic Flow calculation/approximation

1. `EXPLICIT`: The Harmonic flow gets calculated exactly using LSMR based on the edge flows and given cell complex
2. `PINV`: Updates the harmonic flow as mentioned in the paper, based on the update rule:
   $H^{(i+1)} \gets H^{(i)} - \hat{B} \cdot \hat{B}^\dagger \cdot B \cdot C$
3. `OPT`: Instead maintaining the harmonic flow of each step it gets 'recalculated'.
   The low rank matrix factorization has to support fixing vectors of the solution (so far only `L1_GRAD`),
   e.g. Let $F$ be the matrix to be approximated, let $B$ be a matrix, then the task is to find $B'$, $C$ and $C'$
   such that $F \approx \begin{bmatrix}B & B'\end{bmatrix} \cdot\begin{bmatrix}C \\ C'\end{bmatrix}$ where $[ \cdot ]$ denotes the concatenation in the appropriate axes
   and fitting dimension are assumed.
   Then $F$ is the collection of edge flows and $B$ represents the boundary matrix up to that iteration.
   Consequently, $C$ is part of the computation and $H = F - B \cdot C$ is the current approximated harmonic flow

#### Low Rank Approxmiaton

1. `SVD`: The singular value decomposition is the mathematical optimal low rank approximation (under the Frobenius norm)
   into Matrices $U$, $\Sigma$ and $V^\top$.
   Then, $B := U \cdot \Sigma^{1/2}$ and $C = \Sigma^{1/2} \cdot C$.
   Only the first $r$ vectors are returned (in order to be a low-rank approximation and not just a decomposition).
2. `LU`: First $B$ and $C$ are calculated as for `SVD`. Then, a LU-decomposition is performed on $B = L \cdot U$.
   As a result we set $B \gets L$ and $C \gets U \cdot C$
3. `L1_GRAD`: We have 3 desirable properties for $B$ and $C$ for that matter:
   - (`w_approx_F`) Approximation of $F$: This is obvious. We measure the loss with $||F - B \cdot C||_2$
   - (`w_close_discrete`) Close to discrete: Entries of $B$ should be close to $\{-1, 0, 1\}$ to represent a valid boundary
     That loss is measured by as the sum of the each entry's distance to a closest member of $\{-1, 0, 1\}$
   - (`w_boundary_one`) The edge-to-node matrix often denoted with $B_1$ multiplied with a valid polygon-to-edge matrix (here $B$) has to be 0:
     Then `L1_GRAD` performs a gradient descent along the weighted gradients of the corresponding losses.
     The initial values are the same as `SVD`
4. `ICA`: Uses Independent Component Analysis ( see [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA))
5. `Sparse PCA`: Similar to SVD but allows to set Sparsity with `alpha` (see [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.SparsePCA.html))

#### Candidate Selection

1. `BL1`: Chooses the columns of $B$ with the biggest $\ell_1$ norm
2. `CL1`: Choose the columns of $B$ whose corresponding $C$-row has the biggest $\ell_1$ norm
3. `B_CLOSE_DISCRETE`: Chooses the columns of $B$ with lowest sum of entropy of the absolute of the entries
4. `APPROX_F`: Chooses the columns of $B$, that in conjunction with the respective $C$ row, best approximate $F$.
   (Note: In case of the SVD, those are the first columns)
5. `WEIGHTED`: Uses the same loss and weights as `L1_GRAD`.
   Uses the losses of the SVD as normalization factor so that weights are meaningful

#### Discretization Method

1. `RANKED_EDGES`: The chosen vectors assign each edge their corresponding value (strength).
   Then, add the strongest vectors to an initially empty graph until the first (unique) cycle is found.
   Return the 2-cell represented by that cycle
2. `P_WALK`: Similar to `RANKED_EDGES` assigns each edge its strength. A randomly walking agent is simulated and uses the
   strengths of edges as weights for the probabilities. Some edge cases are considered and a confidence value is maintained
   to account for the direction of travel, which is increased if possible.
3. `DISCRETE_FIRST`: Discretizes the values of the vector according to a threshold (closest neighbors to -1, 0 or 1).
   Then these values are used as direction for an directed graph (or no edge if 0).
   If a cycle is found (by NetworkX) in the corresponding graph return the corresponding 2-cell.
   Uses `P_WALK` with the discrete values as fallback

## Known Issues/Limitations

If a `config_NAME.yaml` is changed the intermediate files are not recomputed, but only the missing ones.
On the one hand, it is good to reuse the results.
On the other hand, if iterations or repetitions are increased and previous intermediate results reused
then the end result is not as accurate as desired.
Thus the according files should be deleted or a new group with the modification used.

In order to enable SLURM-Support the resulting paths to intermediate files are less readable than without token replacement.
Additionally, some hard-coded Strings are less intuitive, e.g. "HASH" as a separator instead of "#"

The `n_best` parameter determines the maximum amount of cells that are added in each iteration.
Each 2-cell is only added if it unique, but the same 2-cell might be inferred multiple times.

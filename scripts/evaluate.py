import pprint

import pandas as pd
from snakemake.script import Snakemake

import helpers
from src.commons import DisconnectedGraph
from src.util import seed_int


def fix_smk() -> Snakemake:
    return snakemake


snakemake = fix_smk()

"""
The main script. Executes an experiment class on the corresponding graph/cc
Writes the files for raw error, times, added_cells, etc.
"""

cc_config, experiment, fun_config, _ = helpers.parse_file(snakemake.output[0])
ITERATIONS = snakemake.params.iterations
REPETITIONS = snakemake.params.repetitions


if not isinstance(fun_config, dict) or not all(
    isinstance(k, str) for k in fun_config.keys()
):
    raise ValueError("exp_config must be a dictionary with string keys")

errors_df = pd.DataFrame()
times_df = pd.DataFrame()
cell_lengths_df = pd.DataFrame()
memory_df = pd.DataFrame()
representable_cells = pd.DataFrame()
added_cells = {}


# Using oddd seeds for graph generation and even seeds for execution
#
def run_experiment(set_non_zero_memory):
    global errors_df
    global times_df
    global cell_lengths_df
    global memory_df
    global added_cells
    global representable_cells
    seed = 0
    i = 0
    while i < REPETITIONS:
        try:
            test = experiment(
                {
                    **cc_config,
                    "seed": seed_int(2 * seed),
                    "input": snakemake.input.generator_input,
                },
                ITERATIONS,
                **{k: v for k, v in fun_config.items() if k[0] != "_"},
                seed=seed_int(2 * seed + 1),
                trace_memory=set_non_zero_memory,
            )
        except DisconnectedGraph:
            print("Regenerating. Seed", seed)
            seed += 1
            continue
        test.run()
        i += 1
        seed += 1

        if set_non_zero_memory:
            memory_df[i] = test.get_memory()
        else:
            errors_df[i] = test.get_errors()
            times_df[i] = test.get_times()
            cell_lengths = test.get_cell_lengths()
            # pad to same length
            if len(cell_lengths) > len(cell_lengths_df):
                cell_lengths_df = cell_lengths_df.reindex(
                    range(len(cell_lengths))
                ).fillna(0)
            if len(cell_lengths) < len(cell_lengths_df):
                cell_lengths += [0] * (len(cell_lengths_df) - len(cell_lengths))
            cell_lengths_df[i] = test.get_cell_lengths()
            added_cells[i] = test.get_added_cells()
            memory_df[i] = test.get_memory()  # set to 0s
            representable_cells[i] = test.get_representable_cells()


run_experiment(set_non_zero_memory=snakemake.params.trace_memory)

errors_df.to_csv(snakemake.output.errors)
times_df.to_csv(snakemake.output.times)
cell_lengths_df.to_csv(snakemake.output.cell_lengths)
memory_df.to_csv(snakemake.output.memory)
representable_cells.to_csv(snakemake.output.representable_cells)
with open(snakemake.output.added_cells, "w+") as f:
    f.write(pprint.pformat(added_cells, indent=2))

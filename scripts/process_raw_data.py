import pandas as pd
from snakemake.script import Snakemake


def fix_smk() -> Snakemake:
    return snakemake


snakemake = fix_smk()

"""
Computes the cells to baseline metric
"""


errors = pd.read_csv(snakemake.input.errors, index_col=0)
times = pd.read_csv(snakemake.input.times, index_col=0)
base_errors = pd.read_csv(snakemake.input.base_errors, index_col=0)
base_times = pd.read_csv(snakemake.input.base_times, index_col=0)
random_errors = pd.read_csv(snakemake.input.random_errors, index_col=0)
construction_errors = pd.read_csv(snakemake.input.construction_errors, index_col=0)

## Outputs
relative_performance: pd.DataFrame = (random_errors - errors) / (
    random_errors - base_errors
)
relative_performance.to_csv(snakemake.output.relative_performance)


# Cumulative Times
cumulative_times: pd.DataFrame = times.cumsum(axis=0)
cumulative_times.to_csv(snakemake.output.cumulative_times)

# Naive Error Ratios
naive_error_ratios: pd.DataFrame = errors / base_errors
naive_error_ratios.to_csv(snakemake.output.naive_error_ratios)

# Error Ratios
error_ratios: pd.DataFrame = (errors - construction_errors) / (
    base_errors - construction_errors
)
error_ratios.to_csv(snakemake.output.error_ratios)

# Cells To Baseline
cells_to_baseline = pd.DataFrame()
for compare_n in range(min(snakemake.params.iterations, base_errors.shape[0])):
    for iteration in base_errors.columns:
        # PERF: Is int(iteration) necessary? rather min(..,..) as above?
        if int(iteration) >= snakemake.params.repetitions:
            continue
        base_error = base_errors.iloc[compare_n][iteration]
        min_idx = errors[errors <= base_error][iteration].first_valid_index()
        if compare_n in cells_to_baseline.index:
            cells_to_baseline.at[compare_n, iteration] = min_idx
        else:
            cells_to_baseline = pd.concat(
                (
                    cells_to_baseline,
                    pd.DataFrame({iteration: [min_idx]}, index=[compare_n]),
                )
            )
# cells_to_baseline = cells_to_baseline.astype("Int64")
cells_to_baseline.to_csv(snakemake.output.cells_to_baseline)

# Error Difference
error_difference: pd.DataFrame = errors - base_errors
error_difference.to_csv(snakemake.output.error_difference)

# Time Ratios
time_ratios: pd.DataFrame = times / base_times
time_ratios.to_csv(snakemake.output.time_ratios)

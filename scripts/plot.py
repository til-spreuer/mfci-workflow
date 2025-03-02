from itertools import count
from pprint import pformat

import matplotlib.pyplot as plt
import pandas as pd
from pypdf import PageObject, PdfReader, PdfWriter
from pypdf.annotations import FreeText
from snakemake.script import Snakemake

from helpers import parse_file


def fix_smk() -> Snakemake:
    return snakemake


"""
The script that actually plots the different metrics.
Each plot is written to its own file, which allows easier extraction.
Also, all plots get combined into one file
"""

snakemake = fix_smk()

SMALL_SIZE = 18
MEDIUM_SIZE = 20
BIG_SIZE = 22

plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
plt.rc("axes", titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc("legend", fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc("figure", titlesize=BIG_SIZE)  # fontsize of the figure title
# plt.subplots_adjust(left=0.4, right=0.9, bottom=0.4, top=0.9)
# plt.tight_layout()
"""
Creates the plots of every metric each in its own pdf
Helpful for papers if not every metric is relevant at each point
"""

MAIN_COLOR = "#1f77b4"
BASE_COLOR = "#ff7f0e"
RANDOM_COLOR = "#2ca02c"
CONSTRUCTION_COLOR = "#d62728"
SVD_COLOR = "#9467bd"


main = {}
base = {}
random_inference = {}
none_inference = {}
construction = {}
svd_inference = {}

si = snakemake.input
so = snakemake.output

if snakemake.wildcards.plot_mode[:4] == "mean":
    plot_mean = True
elif snakemake.wildcards.plot_mode[:4] == "each":
    plot_mean = False
else:
    raise NotImplementedError


def _plot(lines, ax, title="", ylabel="Mean", xlabel="Iteration", xlim_left=0):
    plt.sca(ax)
    max_len = 0
    added_labels = []
    for line in lines:
        df, label, color = line[0], "", None
        if len(line) > 1:
            label = line[1]
        if len(line) > 2:
            color = line[2]
        if plot_mean:
            df_agg = df.aggregate(["mean", "std"], axis=1)

            plt.plot(df_agg["mean"], color=color, label=label)
            plt.fill_between(
                x=df_agg.index,
                y1=df_agg["mean"] - df_agg["std"],
                y2=df_agg["mean"] + df_agg["std"],
                alpha=0.2,
                color=color,
            )
        else:
            for _, col in df.T.iterrows():
                if label not in added_labels:
                    plt.plot(col, color=color, linewidth=1, alpha=1, label=label)
                    added_labels.append(label)
                else:
                    plt.plot(col, color=color, linewidth=1, alpha=1)

        plt.legend(
            bbox_to_anchor=(0.5, -0.2),
            loc="upper center",
            ncol=2,
            frameon=False,
        )

        if len(df) > max_len:
            max_len = len(df)

    ax.set_xlim([xlim_left, max_len - 1])  # Set x-axis limits
    ax.grid(True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot(
    lines,
    ax,
    title="",
    ylabel="Mean",
    xlabel="No. added Cells",
    out=None,
    xlim_left=0,
):
    _plot(lines, ax, title, ylabel, xlabel, xlim_left)
    if out is not None:
        _, ax = plt.subplots()
        _plot(lines, ax, title, ylabel, xlabel, xlim_left)
        plt.savefig(out, bbox_inches="tight")
    plt.figure(fig)


def read(file: str) -> pd.DataFrame:
    """
    Short function to read file path that leads to a csv as pd.DataFrame
    """
    return pd.read_csv(file, index_col=0)


fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(20, 20))
fig.subplots_adjust(hspace=1, wspace=0.5)
axs = axs.flatten()
ax_count = count()
plot(
    [
        (read(si.errors), "Main", MAIN_COLOR),
        (read(si.base_errors), "Base", BASE_COLOR),
        (read(si.random_errors), "Random", RANDOM_COLOR),
        (read(si.construction_errors), "Construction", CONSTRUCTION_COLOR),
        (read(si.svd_errors), "SVD", SVD_COLOR),
    ],
    axs[next(ax_count)],
    "Absolute Errors",
    ylabel=r"$||\text{harm}(\mathbf{F})||_2$",
    out=so.absolute_errors,
)

plot(
    [
        (
            read(si.relative_performance),
            "Main",
            MAIN_COLOR,
        ),
    ],
    axs[next(ax_count)],
    "Relative Performance",
    ylabel="Scale",
    out=so.relative_performance,
)

plot(
    [
        (read(si.cumulative_times), "Main", MAIN_COLOR),
        (read(si.base_cumulative_times), "Base", BASE_COLOR),
    ],
    axs[next(ax_count)],
    "Cumulative Times",
    ylabel="Time [s]",
    out=so.cumulative_times,
)

plot(
    [
        (read(si.representable_cells), "Main", MAIN_COLOR),
        (read(si.base_representable_cells), "Base", BASE_COLOR),
        (read(si.random_representable_cells), "Random", RANDOM_COLOR),
    ],
    axs[next(ax_count)],
    "Representable Cells",
    out=so.representable_cells,
    ylabel="No. Cells Added",
)


plot(
    [
        (read(si.naive_error_ratios), "Main / Base", MAIN_COLOR),
    ],
    axs[next(ax_count)],
    "Naive Error Ratios",
    ylabel="Percentage",
    out=so.naive_error_ratios,
)


error_ratios = read(si.error_ratios)
er_min = error_ratios.min().min()
er_max = error_ratios.max().max()

plot(
    [
        (
            error_ratios,
            "Main / Base",
            MAIN_COLOR,
        ),
    ],
    axs[next(ax_count)],
    "Error Ratios (-Construction)",
    ylabel="Percentage",
    out=so.error_ratios_wo_construction,
)

plot(
    [
        (
            read(si.cells_to_baseline),
            "Main",
            MAIN_COLOR,
        ),
        (
            read(si.random_cells_to_baseline),
            "Random",
            RANDOM_COLOR,
        ),
        (
            read(si.base_cells_to_baseline),
            "Base",
            BASE_COLOR,
        ),
    ],
    axs[next(ax_count)],
    "Cells to reach Baseline",
    ylabel="No. added Cells",
    xlabel="No. added Cells (baseline algorithm)",
    out=so.cells_to_baseline,
)

plot(
    [
        (read(si.error_difference), "Main - Base", MAIN_COLOR),
        (read(si.random_error_difference), "Random - Base", RANDOM_COLOR),
    ],
    axs[next(ax_count)],
    "Main - Base Error Difference",
    ylabel=r"Difference in $||\text{harm}(\mathbf{F})||_2$",
    out=so.error_difference,
)


plot(
    [(read(si.time_ratios), "Main / Base", MAIN_COLOR)],
    axs[next(ax_count)],
    "Time Ratios",
    ylabel="Percentage",
)

plot(
    [
        (read(si.cell_lengths), "Main", MAIN_COLOR),
        (read(si.base_cell_lengths), "Base", BASE_COLOR),
        (read(si.random_cell_lengths), "Random", RANDOM_COLOR),
        (read(si.construction_cell_lengths), "Construction", CONSTRUCTION_COLOR),
    ],
    axs[next(ax_count)],
    "Cell Lengths",
    ylabel="Amount",
    xlabel="Cell Lengths",
    out=so.cell_lengths,
    xlim_left=3,
)

plot(
    [
        (read(si.memory), "Main", MAIN_COLOR),
        (read(si.base_memory), "Base", BASE_COLOR),
    ],
    axs[next(ax_count)],
    "Memory Usage",
    out=so.memory,
    ylabel="Memory Blocks",
)

plt.savefig(so.temp_pdf, bbox_inches="tight")


BIGMARGIN = 200  # Top Level Space for comments
SMALLMARGIN = 20  # Page Margin

base_config = {
    "heuristic": snakemake.config[snakemake.params.group]["heuristic"],
    "n_candidates": snakemake.config[snakemake.params.group]["n_candidates"],
    "n_clusters": snakemake.config[snakemake.params.group]["n_clusters"],
}


writer = PdfWriter()
cc_config, experiment, exp_config, approach = parse_file(so.plot)

reader = PdfReader(so.temp_pdf)
page = reader.pages[0]
margin_page = PageObject.create_blank_page(
    width=page.mediabox.width, height=page.mediabox.height + BIGMARGIN
)
margin_page.merge_scaled_page(page, scale=1, over=True, expand=False)
writer.add_page(margin_page)

text_exp = f"{approach}\n"
text_exp += pformat(exp_config, indent=2)
text_exp += f"\nvs\n"
# Add baseline name?
text_exp += pformat(base_config, indent=2)

annotation_exp = FreeText(
    text=text_exp,
    rect=[
        SMALLMARGIN,
        margin_page.mediabox.height - BIGMARGIN + SMALLMARGIN,
        margin_page.mediabox.width // 2 - SMALLMARGIN,
        margin_page.mediabox.height - SMALLMARGIN,
    ],  # type: ignore
    font_size="24pt",
)
text_cc = f"{cc_config["graph_class"]}\n"
text_cc += pformat(cc_config, indent=2)
annotation_cc = FreeText(
    text=text_cc,
    rect=[
        margin_page.mediabox.width // 2 + SMALLMARGIN,
        margin_page.mediabox.height - BIGMARGIN + SMALLMARGIN,
        margin_page.mediabox.width - SMALLMARGIN,
        margin_page.mediabox.height - SMALLMARGIN,
    ],  # type: ignore
    font_size="24pt",
)

writer.add_annotation(page_number=0, annotation=annotation_exp)
writer.add_annotation(page_number=0, annotation=annotation_cc)


for page in writer.pages:
    page.compress_content_streams()

writer.write(so.plot)
writer.close()

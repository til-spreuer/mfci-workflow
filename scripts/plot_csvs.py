import matplotlib.pyplot as plt
import pandas as pd
from snakemake.script import Snakemake


def fix_smk() -> Snakemake:
    return snakemake


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

fig, ax = plt.subplots()
plt.margins(x=0)
ax.grid(True)

try:
    ax.set_xlabel(snakemake.params.xlabel)
except AttributeError:
    pass

try:
    ax.set_ylabel(snakemake.params.ylabel)
except AttributeError:
    pass

try:
    ax.set_ylim(snakemake.params.ylim)
except AttributeError:
    pass

try:
    ax.set_title(snakemake.params.title)
except AttributeError:
    pass

try:
    confidence_alpha = snakemake.params.confidence_alpha
except AttributeError:
    confidence_alpha = 0.2
try:
    legend_ncol = snakemake.params.legend_ncol
except AttributeError:
    legend_ncol = 3

CMAP = None
LEN = None
try:
    CMAP = plt.get_cmap(snakemake.params.cmap)
    LEN = len(snakemake.params.labels)
except AttributeError:
    pass
except ValueError:
    print(f"[WARN]: {snakemake.params.cmap} is no valid cmap. Use default instead")


for ix, (label, csv) in enumerate(zip(snakemake.params.labels, snakemake.input)):
    df = pd.read_csv(csv, index_col=0)
    df_agg = df.aggregate(["mean", "std"], axis=1)
    (line,) = plt.plot(
        df_agg["mean"],
        label=label,
        color=CMAP(ix / (LEN - 1)) if CMAP and LEN else None,
    )
    plt.fill_between(
        x=df_agg.index,
        y1=df_agg["mean"] - df_agg["std"],
        y2=df_agg["mean"] + df_agg["std"],
        alpha=confidence_alpha,
        color=line.get_color(),
    )
try:
    if snakemake.params.flat_line:
        plt.axhline(y=1, color="grey", linestyle="--")
except AttributeError:
    pass
plt.legend(
    bbox_to_anchor=(0.5, -0.2), loc="upper center", ncol=legend_ncol, frameon=False
)
plt.savefig(snakemake.output[0], bbox_inches="tight")

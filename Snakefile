configfile: "config.yaml"


import itertools
import yaml
import scripts.helpers
import os

cfg_dir = os.path.join(workflow.basedir, "configs")
cfgs, baseline_path_of_group = scripts.helpers.prepare_cfgs(cfg_dir, config)

# As those are created by Snakemake do not give user the access to change/break it
# However very useful inside the configuration
taxi_graph_path = "resources/taxi/graph.txt"
taxi_flows_path = "resources/taxi/flows.csv"


def get_runtime(wildcards, attempt):
    """
    Maps the attempt number to a (reasonable) runtime
    """
    if attempt == 1:
        return 10
    elif attempt == 2:
        return 45
    elif attempt == 3:
        return 120

rule paper:
    input:
        "out/noise_robustness.pdf",
        expand("out/exp_vs_pinv/{metric}.pdf", metric=["cumulative_times", "errors"]),
        expand("out/taxi_tradeoffs/{metric}.pdf", metric=["cumulative_times", "errors"]),
        expand(
            "out/lr_comparsions/lr_on_er_{metric}.pdf",
            metric=["errors", "cumulative_times"],
        ),
    shell:
      "echo 'Creation of figures from paper done. Figures have been adapted with Inkscape for the paper'"




rule all:
    input:
        expand(
            "out/plot_combined/plot_{plot_mode}_{cfg_name}.pdf",
            plot_mode=["mean", "each"],
            cfg_name=[name for name, _ in cfgs],
        ),
        "out/noise_robustness.pdf",
        expand("out/taxi_tradeoffs/{metric}.pdf", metric=["cumulative_times", "errors"]),
        expand(
            "out/lr_comparsions/lr_on_er_{metric}.pdf",
            metric=["errors", "cumulative_times"],
        ),
        expand("out/exp_vs_pinv/{metric}.pdf", metric=["cumulative_times", "errors"]),
    localrule: True
    shell:
        "echo 'all plots created'"


rule noise_robustness:
    input:
        expand(
            "out/csvs/noise/HASHER/nTD40/pTD0DOT9/two_cellsTD80/flowsTD64/flow_multTD1/"
            + "flow_addTD0/noise_multTD{noise}/independent_flowsTDTrue/remove_unused_edgesTDFalse/"
            + "HASHmat_fact/"
            # + "candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN1CMMQTw_close_discreteQTCLN1RB/"
            # + "candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/"
            + "candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/"
            + "discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/"
            + "harmonic_method_and_paramsTDPINVTDLBRB/"
            + "lr_method_and_paramsTDSVDTDLBQTrQTCLN16RB/n_bestTD1/n_candidatesTD5/relative_performance.csv",
            noise=["0", "0DOT2", "0DOT4", "0DOT6", "0DOT8", "1"],
        ),
    params:
        labels=[
            r"$\sigma = 0$",
            r"$\sigma = 0.2$",
            r"$\sigma = 0.4$",
            r"$\sigma = 0.6$",
            r"$\sigma = 0.8$",
            r"$\sigma = 1$",
        ],
        xlabel="No. Cells Added",
        ylabel="Relative Performance",
        title="Noise Robustness",
        ylim=[0, 2],
        flat_line=True,
        confidence_alpha=0.1,
        cmap="viridis",
    output:
        "out/noise_robustness.pdf",
    script:
        "scripts/plot_csvs.py"


rule pinv_vs_exp:
    input:
        expand(
            "out/exp_vs_pinv/{metric}.pdf",
            metric=["errors", "cumulative_times"],
        ),
    shell:
        "echo 'created  Harmonic Flow: explicit vs pinv approximation'"


rule _pinv_vs_exp:
    input:
        mf_pinv_1oo8="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/harmonic_method_and_paramsTDPINVTDLBRB/lr_method_and_paramsTDICATDLBQTrQTCLN8RB/n_bestTD1/n_candidatesTD8/{metric}.csv",
        mf_pinv_8oo_minus1="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/harmonic_method_and_paramsTDPINVTDLBRB/lr_method_and_paramsTDICATDLBQTrQTCLN8RB/n_bestTD8/n_candidatesTD-1/{metric}.csv",
        mf_exp_1oo8="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/harmonic_method_and_paramsTDEXPLICITTDLBRB/lr_method_and_paramsTDICATDLBQTrQTCLN8RB/n_bestTD1/n_candidatesTD8/{metric}.csv",
        mf_exp_8oo_minus1="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/harmonic_method_and_paramsTDEXPLICITTDLBRB/lr_method_and_paramsTDICATDLBQTrQTCLN8RB/n_bestTD8/n_candidatesTD-1/{metric}.csv",
        cf_sim="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHcell_flower/heuristicTDSIMILARITY/n_candidatesTD11/n_clustersTD11/{metric}.csv",
        construction="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHconstruction/{metric}.csv"
    params:
        labels=[
          "MFCI (pinv, 1oo8)",
          "MFCI (pinv, 8oo-1)",
          "MFCI (exp, 1oo8)",
          "MFCI (exp, 8oo-1)",
          "SPH",
          "Construction"
        ],
        title="Influence of Harmonic Flow Approximation",
        xlabel="No. added Cells",
        ylabel=lambda wildcards: (
            r"$||$harm$(\mathbf{F})||_2$"
            if wildcards["metric"] == "errors"
            else r"Time [s]"
        ),
        # lambda funtcion to allow {F} notation without wildcards replacement
        legend_ncol=3,
        confidence_alpha=0.1,
    output:
        "out/exp_vs_pinv/{metric}.pdf",
    script:
        "scripts/plot_csvs.py"


rule taxi_tradeoffs:
    input:
        expand("out/taxi_tradeoffs/{metric}.pdf", metric=["errors", "cumulative_times"]),
    shell:
        "echo 'created taxi tradeoffs'"


rule _taxi_tradeoffs:
    input:
        cf_max="out/csvs/default/HASHTAXI/flowsTD128/HASHcell_flower/heuristicTDMAX/n_candidatesTD1/n_clustersTD0/{metric}.csv",
        cf_sim="out/csvs/default/HASHTAXI/flowsTD128/HASHcell_flower/heuristicTDSIMILARITY/n_candidatesTD11/n_clustersTD11/{metric}.csv",
        mf_svd="out/csvs/default/HASHTAXI/flowsTD128/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDP_WALKTDLBRB/harmonic_method_and_paramsTDPINVTDLBRB/lr_method_and_paramsTDSVDTDLBQTrQTCLN16RB/n_bestTD1/n_candidatesTD-1/{metric}.csv",
        mf_ica="out/csvs/default/HASHTAXI/flowsTD128/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDP_WALKTDLBRB/harmonic_method_and_paramsTDPINVTDLBRB/lr_method_and_paramsTDICATDLBQTrQTCLN16RB/n_bestTD1/n_candidatesTD-1/{metric}.csv",
        svd="out/csvs/default/HASHTAXI/flowsTD128/HASHsvd/{metric}.csv",
        random="out/csvs/default/HASHTAXI/flowsTD128/HASHrandom_inference/{metric}.csv",
    params:
        labels=[
            "Max Spanning Trees",
            "Similarity Spanning Trees",
            "MFCI (SVD, 1oo-1)",
            "MFCI (ICA, 1oo-1)",
            "SVD",
            "Random",
        ],
        title="Comparison of LR-Methods on Taxi",
        xlabel="No. added Cells",
        ylabel=lambda wildcards: (
            r"$||$harm$(\mathbf{F})||_2$"
            if wildcards["metric"] == "errors"
            else r"Time [s]"
        ),
        # lambda funtcion to allow {F} notation without wildcards replacement
        legend_ncol=3,
        confidence_alpha=0.1,
    output:
        "out/taxi_tradeoffs/{metric}.pdf",
    script:
        "scripts/plot_csvs.py"


rule lr_on_er:
    input:
        expand(
            "out/lr_comparsions/lr_on_er_{metric}.pdf",
            metric=["errors", "cumulative_times"],
        ),
    shell:
        "echo 'created LR comparisons on er'"


rule _lr_on_er:
    input:
        mf_ica_1oo8="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/harmonic_method_and_paramsTDPINVTDLBRB/lr_method_and_paramsTDICATDLBQTrQTCLN8RB/n_bestTD1/n_candidatesTD8/{metric}.csv",
        mf_ica_8oom1="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/harmonic_method_and_paramsTDPINVTDLBRB/lr_method_and_paramsTDICATDLBQTrQTCLN8RB/n_bestTD8/n_candidatesTD-1/{metric}.csv",
        svd="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHsvd/{metric}.csv",
        random="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHrandom_inference/{metric}.csv",
        construction="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHconstruction/{metric}.csv",
        cf_sim="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHcell_flower/heuristicTDSIMILARITY/n_candidatesTD11/n_clustersTD11/{metric}.csv",
    params:
        labels=[
            "mf-ICA-1oo8",
            "mf-ICA-8oo-1",
            "svd",
            "random",
            "construction",
            "cf_sim",
        ],
        title="Comparison of LR-Methods on ER",
        xlabel="No. added Cells",
        ylabel=lambda wildcards: (
            r"$||$harm$(\mathbf{F})||_2$"
            if wildcards["metric"] == "errors"
            else r"Time [s]"
        ),
        # lambda funtcion to allow {F} notation without wildcards replacement
        legend_ncol=3,
        confidence_alpha=0.1,
    output:
        "out/lr_comparsions/lr_on_er_{metric}.pdf",
    script:
        "scripts/plot_csvs.py"


rule compare_lr_all:
    input:
        expand(
            "out/lr_comparsions_all/er_{best}oo{candidate}_{metric}.pdf",
            best=[1, 8],
            candidate=[8, -1],
            metric=["errors", "cumulative_times"],
        ),
    shell:
        "echo 'LR comparisons created'"


rule _compare_lr_er_all:
    input:
        mf_ica="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/harmonic_method_and_paramsTDPINVTDLBRB/lr_method_and_paramsTDICATDLBQTrQTCLN8RB/n_bestTD{best}/n_candidatesTD{candidate}/{metric}.csv",
        mf_spca="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/harmonic_method_and_paramsTDPINVTDLBRB/lr_method_and_paramsTDSPCATDLBQTrQTCLN8CMMQTalphaQTCLN5RB/n_bestTD{best}/n_candidatesTD{candidate}/{metric}.csv",
        mf_grad_40="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/harmonic_method_and_paramsTDPINVTDLBRB/lr_method_and_paramsTDL1_GRADTDLBQTrQTCLN8CMMQTw_boundary_oneQTCLN1CMMQTw_approx_FQTCLN1CMMQTw_close_discreteQTCLN1CMMQTiterationsQTCLN40RB/n_bestTD{best}/n_candidatesTD{candidate}/{metric}.csv",
        mf_grad_200="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/harmonic_method_and_paramsTDPINVTDLBRB/lr_method_and_paramsTDL1_GRADTDLBQTrQTCLN8CMMQTw_boundary_oneQTCLN1CMMQTw_approx_FQTCLN1CMMQTw_close_discreteQTCLN1CMMQTiterationsQTCLN200RB/n_bestTD{best}/n_candidatesTD{candidate}/{metric}.csv",
        mf_svd="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/harmonic_method_and_paramsTDPINVTDLBRB/lr_method_and_paramsTDSVDTDLBQTrQTCLN8RB/n_bestTD{best}/n_candidatesTD{candidate}/{metric}.csv",
        mf_lu="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHmat_fact/candidate_selection_and_paramsTDWEIGHTEDTDLBQTw_approx_FQTCLN1CMMQTw_boundary_oneQTCLN0CMMQTw_close_discreteQTCLN0RB/discretizing_method_and_paramsTDRANKED_EDGESTDLBRB/harmonic_method_and_paramsTDPINVTDLBRB/lr_method_and_paramsTDLUTDLBQTrQTCLN8RB/n_bestTD{best}/n_candidatesTD{candidate}/{metric}.csv",
        random="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHrandom_inference/{metric}.csv",
        construction="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHconstruction/{metric}.csv",
        svd="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHsvd/{metric}.csv",
        cf_sim="out/csvs/less-iterations/HASHER/nTD40/pTD0DOT9/two_cellsTD50/flowsTD64/flow_multTD1/flow_addTD0/noise_multTD0DOT3/independent_flowsTDTrue/remove_unused_edgesTDFalse/HASHcell_flower/heuristicTDSIMILARITY/n_candidatesTD11/n_clustersTD11/{metric}.csv",
    params:
        labels=[
            "mf-ICA",
            "mf-SPCA (alpha=5)",
            "mf-Grad (40)",
            "mf-Grad (200)",
            "mf-svd",
            "mf-lu",
            "random",
            "construction",
            "svd",
            "cf_sim",
        ],
        title="Comparison of LR-Methods on ER",
        xlabel="No. added Cells",
        ylabel=lambda wildcards: (
            r"$||$harm$(\mathbf{F})||_2$"
            if wildcards["metric"] == "errors"
            else r"Time [s]"
        ),
        # lambda funtcion to allow {F} notation without wildcards replacement
        legend_ncol=3,
        confidence_alpha=0.01,
    output:
        "out/lr_comparsions_all/er_{best}oo{candidate,-?\\d+}_{metric}.pdf",
    script:
        "scripts/plot_csvs.py"


for cfg_name, cfg in cfgs:
    for plot_mode in ["each", "mean"]:

        rule:
            name:
                f"combine_{plot_mode}_{cfg_name}"
            input:
                f"out/plot_combined/plot_{plot_mode}_{cfg_name}.pdf",
            run:
                print(f"Plots combined ({plot_mode}, {cfg_name})")


for cfg_name, cfg in cfgs:

    rule:
        name:
            f"_combine_{cfg_name}"
        input:
            list(
                map(
                    lambda x: f"out/plot_single/{cfg['group']}/"
                    + x
                    + "/plot_{plot_mode}.pdf",
                    scripts.helpers.get_paths(cfg),
                )
            ),
        output:
            "out/plot_combined/plot_{plot_mode}_" + f"{cfg_name}.pdf",
        script:
            "scripts/combine_pdfs.py"


for cfg_name, cfg in cfgs:
    for plot_mode in ["each", "mean"]:

        rule:
            name:
                f"plot_{plot_mode}_{cfg_name}"
            input:
                list(
                    map(
                        lambda x: f"out/plot_single/{cfg['group']}/"
                        + x
                        + f"/plot_{plot_mode}.pdf",
                        scripts.helpers.get_paths(cfg),
                    )
                ),
            localrule: True
            shell:
                "echo 'raw plots created. OK'"


for cfg_name, cfg in cfgs:
    for graph_class in cfg["graph_class"]:
        if not cfg["graph_class"][graph_class].get("_enabled", True):
            continue

        rule:
            name:
                f"_plot_{cfg_name}_{graph_class}"
            input:
                # Main
                relative_performance=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/relative_performance.csv",
                cell_lengths=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/cell_lengths.csv",
                cells_to_baseline=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/cells_to_baseline.csv",
                cumulative_times=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/cumulative_times.csv",
                error_difference=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/error_difference.csv",
                error_ratios=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/error_ratios.csv",
                errors=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/errors.csv",
                memory=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/memory.csv",
                naive_error_ratios=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/naive_error_ratios.csv",
                representable_cells=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/representable_cells.csv",
                time_ratios=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/time_ratios.csv",
                times=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/times.csv",
                # Baseline
                base_cell_lengths=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHcell_flower"
                + baseline_path_of_group[cfg["group"]]
                + "/cell_lengths.csv",
                base_cells_to_baseline=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHcell_flower"
                + baseline_path_of_group[cfg["group"]]
                + "/cells_to_baseline.csv",
                base_cumulative_times=f"out/csvs/{cfg['group']}/"
                + "{graph_config}"
                + "/HASHcell_flower"
                + baseline_path_of_group[cfg["group"]]
                + "/cumulative_times.csv",
                base_errors=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHcell_flower"
                + baseline_path_of_group[cfg["group"]]
                + "/errors.csv",
                base_memory=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHcell_flower"
                + baseline_path_of_group[cfg["group"]]
                + "/memory.csv",
                base_representable_cells=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHcell_flower"
                + baseline_path_of_group[cfg["group"]]
                + "/representable_cells.csv",
                base_times=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHcell_flower"
                + baseline_path_of_group[cfg["group"]]
                + "/times.csv",
                #Random
                random_cell_lengths=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHrandom_inference/cell_lengths.csv",
                random_cells_to_baseline=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHrandom_inference/cells_to_baseline.csv",
                random_error_difference=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHrandom_inference/error_difference.csv",
                random_errors=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHrandom_inference/errors.csv",
                random_representable_cells=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHrandom_inference/representable_cells.csv",
                # Construction
                construction_cell_lengths=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHconstruction/cell_lengths.csv",
                construction_errors=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHconstruction/errors.csv",
                # Misc
                none_errors=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHnone_inference/errors.csv",
                svd_errors=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHsvd/errors.csv",
            output:
                absolute_errors=f"out/plot_single/{cfg['group']}"
                + f"/{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/absolute_errors_{{plot_mode}}.pdf",
                cell_lengths=f"out/plot_single/{cfg['group']}"
                + f"/{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/cell_lengths_{{plot_mode}}.pdf",
                cells_to_baseline=f"out/plot_single/{cfg['group']}"
                + f"/{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/cells_to_baseline_{{plot_mode}}.pdf",
                cumulative_times=f"out/plot_single/{cfg['group']}"
                + f"/{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/cum_times_{{plot_mode}}.pdf",
                error_difference=f"out/plot_single/{cfg['group']}"
                + f"/{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/error_difference_{{plot_mode}}.pdf",
                error_ratios_wo_construction=f"out/plot_single/{cfg['group']}"
                + f"/{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/error_ratios_wo_construction_{{plot_mode}}.pdf",
                memory=f"out/plot_single/{cfg['group']}"
                + f"/{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/mem_usage_{{plot_mode}}.pdf",
                naive_error_ratios=f"out/plot_single/{cfg['group']}"
                + f"/{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/naive_error_ratios_{{plot_mode}}.pdf",
                plot=f"out/plot_single/{cfg['group']}"
                + f"/{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/plot_{{plot_mode}}.pdf",
                relative_performance=f"out/plot_single/{cfg['group']}"
                + f"/{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/relative_performance_{{plot_mode}}.pdf",
                representable_cells=f"out/plot_single/{cfg['group']}"
                + f"/{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/representable_cells_{{plot_mode}}.pdf",
                temp_pdf=temp(
                    f"out/plot_single/{cfg['group']}"
                    + f"/{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/temp_pdf_{{plot_mode}}.pdf"
                ),
            params:
                group=cfg["group"],
            script:
                "scripts/plot.py"


def intermediate_files():
    intermediate_files_list = []
    for cfg_name, cfg in cfgs:
        intermediate_files_list += (
            expand(
                list(
                    map(
                        lambda x: f"out/csvs/{cfg['group']}/" + x + "/{metric}.csv",
                        scripts.helpers.get_paths(cfg),
                    )
                ),
                metric=[
                    "relative_performance",
                    "cumulative_times",
                    "naive_error_ratios",
                    "error_ratios",
                    "cells_to_baseline",
                    "error_difference",
                    "time_ratios",
                ],
            ),
        )
    return intermediate_files_list


rule intermediate:
    input:
        intermediate_files(),
    localrule: True
    shell:
        "echo 'raw files processed to intermediate csvs'"


for cfg_name, cfg in cfgs:

    rule:
        name:
            f"intermediate_{cfg_name}"
        input:
            expand(
                list(
                    map(
                        lambda x: f"out/csvs/{cfg['group']}/" + x + "/{metric}.csv",
                        scripts.helpers.get_paths(cfg),
                    )
                ),
                metric=[
                    "relative_performance",
                    "cumulative_times",
                    "naive_error_ratios",
                    "error_ratios",
                    "cells_to_baseline",
                    "error_difference",
                    "time_ratios",
                ],
            ),
        localrule: True
        shell:
            "echo 'raw files processed to intermediate csvs'"


# NOTE: Has to set --allow-ambiguity
# NOTE: Due to wildcard capturing one cfg provides for graph of enabled graph_class
# And not only the fitting params
for cfg_name, cfg in cfgs:
    # Ensure that cfg only provides for the right graph_class
    for graph_class in cfg["graph_class"]:
        if not cfg["graph_class"][graph_class].get("_enabled", True):
            continue

        rule:
            name:
                f"_intermediate_{cfg_name}_{graph_class}"
            input:
                errors=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/errors.csv",
                times=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/{approach}/times.csv",
                base_errors=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHcell_flower"
                + baseline_path_of_group[cfg["group"]]
                + "/errors.csv",
                base_times=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHcell_flower"
                + baseline_path_of_group[cfg["group"]]
                + "/times.csv",
                random_errors=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHrandom_inference/errors.csv",
                construction_errors=f"out/csvs/{cfg['group']}/"
                + "{graph_config}/HASHconstruction/errors.csv",
            params:
                iterations=config[cfg["group"]]["iterations"],
                repetitions=config[cfg["group"]]["repetitions"],
            output:
                relative_performance=f"out/csvs/{cfg['group']}/"
                + f"{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/relative_performance.csv",
                cumulative_times=f"out/csvs/{cfg['group']}/"
                + f"{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/cumulative_times.csv",
                naive_error_ratios=f"out/csvs/{cfg['group']}/"
                + f"{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/naive_error_ratios.csv",
                error_ratios=f"out/csvs/{cfg['group']}/"
                + f"{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/error_ratios.csv",
                cells_to_baseline=f"out/csvs/{cfg['group']}/"
                + f"{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/cells_to_baseline.csv",
                error_difference=f"out/csvs/{cfg['group']}/"
                + f"{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/error_difference.csv",
                time_ratios=f"out/csvs/{cfg['group']}/"
                + f"{{graph_config,HASH{graph_class}.*}}/{{approach,HASH.*}}/time_ratios.csv",
            script:
                "scripts/process_raw_data.py"


def raw_files():
    raw_files_list = []
    for cfg_name, cfg in cfgs:
        raw_files_list += (
            expand(
                list(
                    map(
                        lambda x: f"out/csvs/{cfg['group']}/" + x + "/{metric}",
                        scripts.helpers.get_paths(cfg),
                    )
                ),
                metric=[
                    "errors.csv",
                    "times.csv",
                    "cell_lengths.csv",
                    "added_cells.txt",
                    "representable_cells.csv",
                ],
            ),
        )
    return raw_files_list


rule raw:
    input:
        raw_files(),
    localrule: True
    shell:
        "echo 'produced raw files'"


for cfg_name, cfg in cfgs:

    rule:
        name:
            f"raw_{cfg_name}"
        input:
            expand(
                list(
                    map(
                        lambda x: f"out/csvs/{cfg['group']}/" + x + "/{metric}",
                        scripts.helpers.get_paths(cfg),
                    )
                ),
                metric=[
                    "errors.csv",
                    "times.csv",
                    "cell_lengths.csv",
                    "added_cells.txt",
                    "representable_cells.csv",
                ],
            ),
        localrule: True
        shell:
            "echo 'raw files created. OK'"


def input_for_generator(name):
    if name.upper() == "TAXI":
        return [taxi_graph_path, taxi_flows_path]
    # synthetic set -> data is created when needed
    return []


for cfg_name, cfg in cfgs:
    for generator in cfg["graph_class"]:

        rule:
            name:
                f"_raw_{generator}_{cfg_name}"
            input:
                generator_input=input_for_generator(generator),
            output:
                errors=f"out/csvs/{cfg['group']}/"
                + "{delimiter,[^/]+}"
                + generator
                + "{_,.*/}errors.csv",
                times=f"out/csvs/{cfg['group']}/"
                + "{delimiter,[^/]+}"
                + generator
                + "{_,.*/}times.csv",
                cell_lengths=f"out/csvs/{cfg['group']}/"
                + "{delimiter,[^/]+}"
                + generator
                + "{_,.*/}cell_lengths.csv",
                memory=f"out/csvs/{cfg['group']}/"
                + "{delimiter,[^/]+}"
                + generator
                + "{_,.*/}memory.csv",
                added_cells=f"out/csvs/{cfg['group']}/"
                + "{delimiter,[^/]+}"
                + generator
                + "{_,.*/}added_cells.txt",
                representable_cells=f"out/csvs/{cfg['group']}/"
                + "{delimiter,[^/]+}"
                + generator
                + "{_,.*/}representable_cells.csv",
            resources:
                runtime=get_runtime,
            params:
                trace_memory=config[cfg["group"]]["trace_memory"],
                iterations=config[cfg["group"]]["iterations"],
                repetitions=config[cfg["group"]]["repetitions"],
            script:
                "scripts/evaluate.py"


# Rules below here from https://github.com/josefhoppe/edge-flow-cell-complexes
rule get_taxi_dataset:
    output:
        "resources/taxi/Manhattan-taxi-trajectories.tar.gz",
    localrule: True
    shell:
        "gdown 1o6bBC7m9IMYQ1OdCBWjLfMw6MXQPmv9J -O {output}"


rule extract_taxi_dataset:
    input:
        "resources/taxi/Manhattan-taxi-trajectories.tar.gz",
    output:
        expand(
            "resources/taxi/{file}",
            file=[
                "README.txt",
                "neighborhoods.txt",
                "medallions.txt",
                "Manhattan-taxi-trajectories.txt",
            ],
        ),
    localrule: True
    shell:
        "tar -xzf {input} -C resources/taxi --strip-components 1"


rule process_taxi_data:
    input:
        "resources/taxi/Manhattan-taxi-trajectories.txt",
    output:
        taxi_graph_path,
        taxi_flows_path,
    script:
        "scripts/process_taxi.py"

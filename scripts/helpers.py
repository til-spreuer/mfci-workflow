import ast
import os
import itertools
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml


sys.path.append(str(Path(__file__).parent.absolute()))
from experiment import approach_to_experiment, Experiment
from src.commons import GraphClass

with open(str(Path(__file__).parent.parent.absolute()) + "/config.yaml") as f:
    global_config = yaml.safe_load(f)

"""
A collection of helpful functions that are only used for the workflow
and thus, do not belong into lib/util
"""


"""
Used for replacing some characters not working on the HPC Cluster
"""
# FIX: Some use replced symbols, some originals
FORBIDDEN_SYMBOL_MAP = [
    ("{", "LB"),
    ("}", "RB"),
    ("#", "HASH"),
    ("~", "TD"),
    ("'", "QT"),
    (":", "CLN"),
    (".", "DOT"),
    (",", "CMM"),
]


def prepare_cfgs(cfg_dir, global_config):
    for group in global_config:
        if group == "default":
            continue
        for key, value in global_config["default"].items():
            if key not in global_config[group]:
                global_config[group][key] = value

    cfgs = []
    for cfg_name in os.listdir(cfg_dir):
        if cfg_name.endswith(".yaml") and cfg_name.startswith("config_"):
            cfg_path = os.path.join(cfg_dir, cfg_name)
            try:
                with open(cfg_path, "r") as file:
                    cfg = yaml.safe_load(file)
                group = cfg.get("group", "default")
                if group not in global_config:
                    print(
                        f"[WARN]: {group} is no valid group in config.yaml. Using default instead"
                    )
                    group = "default"
                    cfg["group"] = "default"

                # Inherit values from default

                n_clusters = global_config[group]["n_clusters"]
                heuristic = global_config[group]["heuristic"]
                n_candidates = global_config[group]["n_candidates"]
                # Ensure Baseline for every graph
                if "cell_flower" not in cfg["inference"]:
                    cfg["inference"]["cell_flower"] = {
                        "heuristic": [heuristic],
                        "n_clusters": [n_clusters],
                        "n_candidates": [n_candidates],
                    }
                else:
                    cell_flower_cfg = cfg["inference"]["cell_flower"]
                    if heuristic not in cell_flower_cfg:
                        cell_flower_cfg["heuristic"].append(heuristic)
                    if n_candidates not in cell_flower_cfg:
                        cell_flower_cfg["n_candidates"].append(n_candidates)
                    if n_clusters not in cell_flower_cfg:
                        cell_flower_cfg["n_clusters"].append(n_clusters)
                # Ensure Misc inferences
                if "none_inference" not in cfg["inference"]:
                    cfg["inference"]["none_inference"] = {"_dummy": "_"}
                if "construction" not in cfg["inference"]:
                    cfg["inference"]["construction"] = {"_dummy": "_"}
                if "random_inference" not in cfg["inference"]:
                    cfg["inference"]["random_inference"] = {"_dummy": "_"}
                if "svd" not in cfg["inference"]:
                    cfg["inference"]["svd"] = {"_dummy": "_"}

                cfgs.append(
                    (cfg_name.removesuffix(".yaml").removeprefix("config_"), cfg)
                )

            except yaml.YAMLError as e:
                print(
                    f"{cfg_name} could not be loaded with error {e}. Are you sure it is formatted right?"
                )

    baseline_path_of_group = {}
    for group in global_config:
        baseline_path_of_group[group] = config_to_file_name(
            {
                "heuristic": global_config[group]["heuristic"],
                "n_clusters": global_config[group]["n_clusters"],
                "n_candidates": global_config[group]["n_candidates"],
            },  # Glboal config can also contain iterations, repetitions
            approach="cell_flower",
        )
    return cfgs, baseline_path_of_group


def config_to_file_renaming(
    fun_config: Dict[str, Any], approach: str
) -> Dict[str, Any]:
    """
    The given config for a an experiment class gets some names substituted
    to be bettter humanly readable on the file system
    Also see in the experiment.py

    Parameters
    ----------
    fun_config : Dict[str, Any]
        The paramters of the approach experiment class as a dictionary

    approach : str
        The string identifier of an experiment class

    Returns
    -------
    Dict[str, Any]
        The paramters with more readable names
    """
    return approach_to_experiment(approach).config_to_file_rename(fun_config)


def file_to_config_renaming(fun_config, approach):
    """
    The given config for a an experiment class gets some names resubstituted
    to be machine conform again
    Also see in the experiment.py

    Parameters
    ----------
    fun_config : Dict[str, Any]
        The paramters of the approach experiment class as a dictionary

    approach : str
        The string identifier of an experiment class

    Returns
    -------
    Dict[str, Any]
        The paramters machine conform again
    """
    return approach_to_experiment(approach).file_to_config_rename(fun_config)


def _convert_str(s: str) -> int | float | bool | str:
    """
    A small function to convert a string into int, float or bool reasonably
    String literal of "True" or "False" will be interperted as bool
    Especially useful for read yaml files

    Parameters
    ----------
    s : str
        The string to be converted

    Returns
    -------
    int | float | bool | str
        The most intuiitive conversion
    """
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s == "True":
        return True
    elif s == "False":
        return False
    else:
        return s


def parse_file(
    file: str,
) -> tuple[Dict[Any, Any], type[Experiment], Dict[str, Any] | None, str]:
    """
    Given a file path extracts the information about the which graph generator
    was used, what the paramters for the graph generator were (nodes, flows, etc.),
    which experiment class is used (cell_flower, svd, etc.),
    the paramters for the experiment class (n_clusters, n_candidates, etc.)

    Parameters
    ----------
    file : str
        The path to be parsed

    Returns
    Dict[Any, Any]
        The options of the graph generator

    Experiment
        The experiment class

    Dict[str, Any] | None
        The options of the experiment class

    str
        The string identifier of the experiment class
    """
    file = path_resubstitute_symbols(file)
    splitted = file.split("#")
    assert len(splitted) == 3, f"# Used somewhere forbidden in file {file}"
    _, cc_part, fun_part = splitted
    cc_options = cc_part.split("/")
    cc_config = {}
    cc_config["graph_class"] = GraphClass[cc_options[0].upper()]
    for cc_option in cc_options[1:-1]:  # first is generator, last is empty
        try:
            pos = cc_option.index("~")
        except ValueError:
            raise Exception("Symbol Replacement not as expected")
        cc_config[cc_option[:pos]] = _convert_str(cc_option[pos + 1 :])

    fun_options = fun_part.split("/")
    approach = fun_options[0]

    experiment = approach_to_experiment(approach)

    exp_config = {}
    for fun_option in fun_options[1:]:
        values = fun_option.split("~")
        if len(values) == 2:
            parameter, value = values
            exp_config[parameter] = _convert_str(value)
        elif len(values) == 3:
            parameter, method, kwds = values
            # rather than json because of "/' issue. Still secure
            kwds = ast.literal_eval(kwds)
            exp_config[parameter] = (method, kwds)
    exp_config = approach_to_experiment(approach).file_to_config_rename(exp_config)
    return cc_config, experiment, exp_config, approach


def sanity_check_path(file):
    """
    Given a file path checks whether it fulfills the sanity conditions
    of the corresponding Experiment class,
    e.g. n_clusters > 0 for cell_flower with SIMILARITY heuristic
    Also see sanity_check of classes in experiment.py

    Parameters
    ----------
    file : str
        The path to check

    Returns
    -------
    bool
        True if filepath represents valid graph/experiment combination
        False otherwise
    """
    file = path_resubstitute_symbols(file)
    cc_config, experiment, exp_config, _ = parse_file(file)
    value = experiment.sanity_check(cc_config, exp_config)
    return value


def config_to_file_name(config: Dict[str, Any], approach: str) -> str:
    """
    Given the config, i.e. param dict, and the str identifier of the experiment class,
    creates the part of the file name.
    Makes the renaming as described in config_to_file_name/experiment renaming
    """
    config = approach_to_experiment(approach).config_to_file_rename(config)
    file = ""
    for name, value in sorted(config.items()):
        if name == "_dummy":
            continue

        if isinstance(value, dict):
            method, method_config = list(value.items())[0]
            if "/" in str(method_config):
                raise ValueError("/ not allowed as symbol in config")
            if "#" in str(method_config):
                raise ValueError("# not allowed as symbol in config")
            if "TD" in str(method_config):
                raise ValueError("TD not allowed as symbol in config")
            file += f"/{name}TD{method}TD{str(method_config).strip().replace(" ", "")}"
        else:  # constant
            file += f"/{name}TD{value}"
    file = path_substitute_symbols(file)
    return file


def get_fun_conf_paths(config: Dict[str, Any]) -> list[str]:
    """
    Given the a config like config.yaml returns all possible combinations
    of paramdicts for different approaches to solve cell inference
    encoded as path/str

    Parameters
    ----------
    config : Dict[str, Any]
        A dict of structure like the config.yaml file

    Return
    ------
    list[str]
        A list of all possible function configuration as paths
    """

    def approach_fun_conf_paths(approach, options):
        expanded_configs = expand_config(options)
        return [
            config_to_file_name(expanded_config, approach)
            for expanded_config in expanded_configs
        ]

    file_paths = []
    for approach, options in config["inference"].items():
        if approach[0] == "_":
            continue
        file_paths += list(
            map(
                lambda f: "#" + approach + f,
                approach_fun_conf_paths(approach, options),
            )
        )
    file_paths = list(map(lambda f: path_substitute_symbols(f), file_paths))
    return file_paths


def get_graph_paths(config):
    """
    Given the a config like config.yaml returns all possible combinations
    of paramdicts for graph/cc generation
    encoded as path/str

    Parameters
    ----------
    config : Dict[str, Any]
        A dict of structure like the config.yaml file

    Return
    ------
    list[str]
        A list of all possible graph configurations as paths
    """

    def str_concat(s, list_of_s):
        return [
            "/".join(combination) for combination in itertools.product([s], list_of_s)
        ]

    graph_paths = [
        str_concat(
            "#" + generator,
            [
                "/".join(s)
                for s in itertools.product(
                    *[
                        [
                            f"{name}TD{value}"
                            for value in config["graph_class"][generator][name]
                        ]
                        for name in config["graph_class"][generator]
                        if name[0] != "_"
                    ]
                )
            ],
        )
        for generator in config["graph_class"]
        if (
            "_enabled" not in config["graph_class"][generator]
            or config["graph_class"][generator]["_enabled"]
        )
    ]
    graph_paths = [item for sublist in graph_paths for item in sublist]
    graph_paths = list(map(lambda f: path_substitute_symbols(f), graph_paths))
    return graph_paths


def get_paths(config):
    """
    Given the a config like config.yaml returns all possible combinations
    graph_paths and file_paths
    i.e. every possible combinations of algorithm/cc that should be tested
    encoded as path/str

    Parameters
    ----------
    config : Dict[str, Any]
        A dict of structure like the config.yaml file

    Return
    ------
    list[str]
        A list of all possible file paths
    """
    graph_paths = get_graph_paths(config)
    file_paths = get_fun_conf_paths(config)
    combined = list(
        map(lambda y: "/".join(y), itertools.product(graph_paths, file_paths)),
    )

    return list(filter(sanity_check_path, combined))


# Config Expansion


def dict_product(dicts):
    """
    From https://stackoverflow.com/a/40623158
    >> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def expand_config(inference_approach_config: Dict[str, Any]) -> list[Dict[str, Any]]:
    """
    Given the config for wich inferences to use (config.yaml)
    gives the product of all possible combinations with consideration
    of lists and dictionaries

    Parameters
    ----------
    inference_approach_config : Dict[str, Any]
        the config under key inference of config.yaml

    Returns
    -------
    dict[str, Any]
        Every possible combination of paramters from the config
        (no sanity checks yet)
    """

    def combine_list_dict_grids(list_level, dict_level):
        """
        Similar to product but expects list_level to be a simple paramter dict,
        e.g. {'a': 0, 'b': 1}.
        dict_level is more involved. A list with tuples of tuples of the form
        [ ((METHOD1 as str, ENUM.A.1 as int, PARAMDICT.A.1), (METHOD2, ENUM.A.2, PARAMDICT.A.2)),
          ((METHOD1 as str, ENUM.B.2 as int, PARAMDICT.B.1), (METHOD2, ENUM.B.2, PARAMDICT.B.1)), ].
        and constructs every possible paramdict from there

        Just combination crunciation
        """
        res = []
        for ls in list_level:
            for d in dict_level:
                tmp = deepcopy(ls)
                for method in d:
                    tmp[method[0]] = {method[1]: method[2]}
                res.append(tmp)
        return res

    if None in inference_approach_config.values():
        return []

    list_level = {
        key: ls for key, ls in inference_approach_config.items() if isinstance(ls, list)
    }
    list_product = list(dict_product(list_level))

    dict_level = []
    methods_count = 0
    for parameter, value in inference_approach_config.items():
        if isinstance(value, dict):
            methods_count += 1
            for method, options in value.items():
                if options is None:
                    continue
                try:
                    for option in dict_product(options):
                        dict_level.append((parameter, method, option))
                except TypeError:
                    # combination with None parameter skipped
                    pass

    def pairwise_different(ls):
        ls = [i[0] for i in ls]
        return len(ls) == len(list(set(ls)))

    expanded_dict_level = itertools.combinations(dict_level, methods_count)
    expanded_dict_level = list(filter(pairwise_different, expanded_dict_level))
    return combine_list_dict_grids(list_product, expanded_dict_level)


# File Renaming


def path_substitute_symbols(path: str) -> str:
    """
    Replaces forbidden symbols with their substitutes

    Parameters
    ----------
    path: str
        A string usually representing a path to be made safe for HPC execution


    Returns
    -------
    str:
        The modified path
    """
    for original, replacement in FORBIDDEN_SYMBOL_MAP:
        path = path.replace(original, replacement)
    return path


def path_resubstitute_symbols(path: str) -> str:
    """
    Makes a resubstitution of the originally forbidden symbols.
    Keep in mind that nowhere is tracked, whether the now present
    symbol combination was replaced or like this in its original
    form. As a result do not use the substitution symbols either

    Parameters
    ----------
    path: str
        A string usually representing a path to be made safe for HPC execution


    Returns
    -------
    str:
        The modified path
    """
    for original, replacement in FORBIDDEN_SYMBOL_MAP:
        path = path.replace(replacement, original)
    return path

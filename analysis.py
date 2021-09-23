#!/usr/bin/env python3
"""
Analyze the results

./analysis.py | tee significance_tests.txt
"""
import os
import sys
import yaml
import pathlib
import collections
import numpy as np
import matplotlib.pyplot as plt

from absl import app
from absl import flags
from scipy import stats
from matplotlib.ticker import MaxNLocator

from pool import run_job_pool
from pickle_data import load_pickle, save_pickle

FLAGS = flags.FLAGS

flags.DEFINE_integer("jobs", 0, "Number of jobs to use for processing files (0 == number of cores)")
flags.DEFINE_bool("paper", False, "Whether to make paper-version plots (e.g. .pdf not .png), outputs to result_plots_paper")
flags.DEFINE_enum("metric", "accuracy", ["accuracy"], "Which metric to plot")
flags.DEFINE_enum("xaxis", "domains", ["uid", "weight", "domains"], "What to use as the x axis")
flags.DEFINE_bool("error_bars_over_runs_not_users", True, "How to compute error bars")

# Use nice names for the plots/tables
nice_method_names = {
    "none": "No Adaptation",
    "upper": "Train on Target",
    "codats": "CoDATS",
    "can": "CAN",

    # No pseudo labeling - Random sampling
    "calda_xs_r": "CALDA-XS,R",
    "calda_in_r": "CALDA-In,R",
    "calda_any_r": "CALDA-Any,R",
    # No pseudo labeling - Hard sampling
    "calda_xs_h": "CALDA-XS,H",
    "calda_in_h": "CALDA-In,H",
    "calda_any_h": "CALDA-Any,H",
    # Pseudo labeling - Random sampling
    "calda_xs_r_p": "CALDA-XS,R,P",
    "calda_in_r_p": "CALDA-In,R,P",
    "calda_any_r_p": "CALDA-Any,R,P",
    # Pseudo labeling - Hard sampling
    "calda_xs_h_p": "CALDA-XS,H,P",
    "calda_in_h_p": "CALDA-In,H,P",
    "calda_any_h_p": "CALDA-Any,H,P",

    # Weak supervision
    "codats_ws": "CoDATS-WS",
    "calda_xs_h_ws": "CALDA-XS,H,WS",
    "calda_any_r_ws": "CALDA-Any,R,WS",

    # Domain generalization
    "codats_dg": "CoDATS-DG",
    "sleep_dg": "Sleep-DG",
    "aflac_dg": "AFLAC-DG",
    "caldg_xs_h": "CALDG-XS,H",
    "caldg_any_r": "CALDG-Any,R",

    # No adversary
    "calda_xs_h_noadv": "CALDA-XS,H,NoAdv",
    "calda_any_r_noadv": "CALDA-Any,R,NoAdv",
}

method_lines = {
    # Approximate bounds
    "none": "-.",
    "upper": "-.",
    # Prior work
    "codats": "--",
}

nice_metric_names = {
    "accuracy": "Accuracy (%)",
    "f1score_macro": "F1 Score (Macro)",
}

dataset_replacements = [
    ("ucihar", "UCI HAR"),
    ("ucihhar", "UCI HHAR"),
    ("wisdm_ar", "WISDM AR"),
    ("wisdm_at", "WISDM AT"),
    ("ninapro_db5_like_myo_noshift", "NinaPro Myo"),
    ("myo", "Myo EMG"),

    # Synthetic datasets
    ("normal_n12_l3_inter0_intra1_5,0,0,0_sine", "Synth InterT 0"),
    ("normal_n12_l3_inter1_intra1_5,0,0,0_sine", "Synth InterT 5"),
    ("normal_n12_l3_inter2_intra1_5,0,0,0_sine", "Synth InterT 10"),
    ("normal_n12_l3_inter0_intra1_0,0.5,0,0_sine", "Synth InterR 0"),
    ("normal_n12_l3_inter1_intra1_0,0.5,0,0_sine", "Synth InterR 0.5"),
    ("normal_n12_l3_inter2_intra1_0,0.5,0,0_sine", "Synth InterR 1.0"),
    ("normal_n12_l3_inter1_intra0_0,0,5,0_sine", "Synth IntraT 0"),
    ("normal_n12_l3_inter1_intra1_0,0,5,0_sine", "Synth IntraT 5"),
    ("normal_n12_l3_inter1_intra2_0,0,5,0_sine", "Synth IntraT 10"),
    ("normal_n12_l3_inter1_intra0_0,0,0,0.5_sine", "Synth IntraR 0"),
    ("normal_n12_l3_inter1_intra1_0,0,0,0.5_sine", "Synth IntraR 0.5"),
    ("normal_n12_l3_inter1_intra2_0,0,0,0.5_sine", "Synth IntraR 1.0"),
]

# Name of problem based on:
# (source_modality_subset, target_modality_subset, shared_modalities)
# Note: (x,) required for single elements to make sure these are tuples
problem_names = {
    (None, None, (0, 1)): "Closed",      # if subset is left as default
    ((0, 1), (0, 1), (0, 1)): "Closed",  # if subset is specified

    ((0,), (0, 1), (0,)): "Open 1",
    ((1,), (1, 0), (0,)): "Open 2",

    ((0, 1), (0,), (0,)): "Partial 1",
    ((1, 0), (1,), (0,)): "Partial 2",

    ((0,), (0,), (0,)): "Single-Modality",
}


def get_tuning_files(dir_name, prefixes):
    """ Get all the hyperparameter evaluation result files """
    files = []
    matching = []

    for prefix in prefixes:
        matching += pathlib.Path(dir_name).glob(prefix+".yaml")

    for m in matching:
        name = m.stem.replace(prefix, "")
        file = str(m)
        files.append((name, file))

    return files


def compute_average(name, data, metric, domain, train_or_valid):
    results = []

    for d in data:
        # Make sure this value exists in the evaluation results .yaml file
        assert "results" in d, \
            "No results in: " + str(d) + " for " + name
        name_of_value = metric+"_task/"+domain+"/"+train_or_valid
        assert name_of_value in d["results"], \
            "No metric value " + name_of_value + " in: " + str(d["results"]) \
            + " for " + name

        result = d["results"][name_of_value]
        results.append(result)

    # There should be 1 or 3 of each; if not, warn
    length = len(results)

    if length != 1 and length != 3:
        print("Warning: number of runs ", length, "(not 1 or 3) for", name,
            file=sys.stderr)

    # ddof=0 is the numpy default, ddof=1 is Pandas' default
    results = np.array(results, dtype=np.float32)
    return results.mean(), results.std(ddof=0), results


def get_method(method, target):
    """
    method="upper" doesn't actually exist since it uses method="none", but
    our upper bound is method="none" without any target domains, so set
    appropriately.
    """
    if method == "none" and target == "":
        method = "upper"

    return method


def _all_stats(name, filename, source_feature_subset, target_feature_subset,
        pickle=True):
    # For speed, if we already loaded this and generated the pickle file,
    # load that instead
    if pickle:
        pickle_filename = "{}.pickle".format(filename)
        results = load_pickle(pickle_filename)

        if results is not None:
            return results

    with open(filename) as f:
        # See: https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
        data = yaml.load(f, Loader=yaml.SafeLoader)

    # Get some of the config
    uid = None
    dataset = None
    method = None
    sources = None
    target = None
    source_modality_subset = None
    target_modality_subset = None
    shared_modalities = None
    similarity_weight = None
    has_results = False

    if len(data) == 0:
        print("Warning: no data in file", filename)
        return {}

    for d in data:
        config = d["config"]

        assert uid is None or config["uid"] == uid, \
            "runs disagree on uid: " \
            + config["uid"] + " vs. " + str(uid)
        uid = config["uid"]

        assert dataset is None or config["dataset"] == dataset, \
            "runs disagree on dataset: " \
            + config["dataset"] + " vs. " + str(dataset)
        dataset = config["dataset"]

        assert sources is None or config["sources"] == sources, \
            "runs disagree on sources: " \
            + config["sources"] + " vs. " + str(sources)
        sources = config["sources"]

        assert target is None or config["target"] == target, \
            "runs disagree on target: " \
            + config["target"] + " vs. " + str(target)
        target = config["target"]

        new_method = get_method(config["method"], target)
        assert method is None or new_method == method, \
            "runs disagree on method: " + new_method + " vs. " + str(method)
        method = new_method

        assert source_modality_subset is None or \
            config["source_modality_subset"] == source_modality_subset, \
            "runs disagree on source_modality_subset: " \
            + config["source_modality_subset"] + " vs. " + str(source_modality_subset)
        source_modality_subset = config["source_modality_subset"]

        assert target_modality_subset is None or \
            config["target_modality_subset"] == target_modality_subset, \
            "runs disagree on target_modality_subset: " \
            + config["target_modality_subset"] + " vs. " + str(target_modality_subset)
        target_modality_subset = config["target_modality_subset"]

        assert shared_modalities is None or \
            config["shared_modalities"] == shared_modalities, \
            "runs disagree on shared_modalities: " \
            + config["shared_modalities"] + " vs. " + str(shared_modalities)
        shared_modalities = config["shared_modalities"]

        assert similarity_weight is None or \
            config["similarity_weight"] == similarity_weight, \
            "runs disagree on similarity_weight: " \
            + config["similarity_weight"] + " vs. " + str(similarity_weight)
        similarity_weight = config["similarity_weight"]

        # Skip if not the right source/target features
        current_source_feature_subset = config["source_feature_subset"]
        current_target_feature_subset = config["target_feature_subset"]

        if source_feature_subset is not None \
                and source_feature_subset != current_source_feature_subset:
            return {}

        if target_feature_subset is not None \
                and target_feature_subset != current_target_feature_subset:
            return {}

        if d["results"] != {}:
            has_results = True

    # Convert to lists of integers
    if source_modality_subset == "":
        source_modality_subset = None
    else:
        source_modality_subset = [int(x) for x in source_modality_subset.split(",")]

    if target_modality_subset == "":
        target_modality_subset = None
    else:
        target_modality_subset = [int(x) for x in target_modality_subset.split(",")]

    shared_modalities = [int(x) for x in shared_modalities.split(",")]

    # Also replace u0, etc. with just "0"
    uid = int(uid.replace("u", ""))

    # Identify problem based on modality subsets and shared modalities
    # Note: convert list to tuple so it's hashable in dictionary
    problem_name = problem_names[(
        tuple(source_modality_subset) if source_modality_subset is not None else None,
        tuple(target_modality_subset) if target_modality_subset is not None else None,
        tuple(shared_modalities)
    )]

    results = {
        "name": name,
        "problem": problem_name,
        "dataset": dataset,
        "method": method,
        "sources": sources,
        "target": target,
        "similarity_weight": similarity_weight,
        "uid": uid,
        # Full data if we need it
        "data": data,
    }

    # For upper bound, there's no target, so instead use the "source" value
    # as the "target" value
    if method == "upper":
        source_or_target = "source"
    else:
        source_or_target = "target"

    # results["results_source_train"] = compute_average(name, data, FLAGS.metric, "source", "training")
    # results["results_source_test"] = compute_average(name, data, FLAGS.metric, "source", "validation")
    # results["results_target_train"] = compute_average(name, data, FLAGS.metric, "target", "training")

    # Would error if we tried computing average with no data
    if not has_results:
        return {}

    results["results_target_test"] = compute_average(name, data, FLAGS.metric, source_or_target, "validation")

    # Cache results
    if pickle:
        save_pickle(pickle_filename, results)

    return results


def all_stats(files, source_feature_subset, target_feature_subset,
        show_progress=True):
    """ Process all files, but since we may have many, many thousands, do it
    with multiple cores by default """
    if FLAGS.jobs == 1:
        results = []

        for name, filename in files:
            results.append(_all_stats(name, filename,
                source_feature_subset, target_feature_subset))
    else:
        commands = []

        for name, filename in files:
            commands.append((name, filename, source_feature_subset,
                target_feature_subset))

        jobs = FLAGS.jobs if FLAGS.jobs != 0 else None
        results = run_job_pool(_all_stats, commands, cores=jobs,
            show_progress=show_progress)

    # Remove empty dictionaries (the "no data" cases)
    results = [r for r in results if r != {}]

    # Sort by name
    results.sort(key=lambda x: x["name"])

    return results


def get_results(run_suffixes, variant_match, source_feature_subset,
        target_feature_subset, tune, folder="results", additional_match="*",
        show_progress=True):
    """ Get the right result files and load them """
    prefixes = [
        "results_"+run_suffix+"_"+variant_match+"-"+additional_match
        for run_suffix in run_suffixes
    ]
    files = get_tuning_files(folder, prefixes)
    results = all_stats(files, source_feature_subset, target_feature_subset,
        show_progress=show_progress)

    # If there's multiple runs with different weights, we want to pick the
    # result from the one with the best validation results, i.e. a grid search
    # for hyperparameter tuning that variable
    if tune:
        results_grouped = collections.defaultdict(lambda: [])

        for result in results:
            results_grouped[(
                result["problem"],
                result["dataset"],
                result["method"],
                result["uid"],
                # uid handles the unique sources/targets
                #result["sources"],
                #result["target"],
                # We don't include this since we want to group by this
                #result["similarity_weight"]
            )].append(result)

        tuned_results = []

        for name, result in results_grouped.items():
            if len(result) > 1:
                # Find the one that had the highest max_accuracy, i.e. on the
                # validation data performed the best
                max_validation_accuracies = []

                for r in result:
                    if len(r["data"]) > 1:
                        print("Warning: found multiple runs for a single weight? using first")

                    max_validation_accuracies.append(r["data"][0]["max_accuracy"])

                max_accuracy = max(max_validation_accuracies)
                index = max_validation_accuracies.index(max_accuracy)
                # We'll use the results from that one
                tuned_results.append(result[index])
            else:
                tuned_results.append(result[0])

        return tuned_results
    else:
        return results


def gen_jitter(length, amount=0.04):
    """ "Dodge" the points slightly on the x axis, so that they don't overlap """
    x = []
    value = -(amount/length)/2

    for i in range(length):
        x.append(value)
        value += amount

    return np.array(x, dtype=np.float32)


def export_legend(legend, dir_name=".", filename="key.pdf", expand=[-5, -5, 5, 5]):
    """ See: https://stackoverflow.com/a/47749903 """
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent()
    bbox = bbox.from_extents(*(bbox.extents + np.array(expand)))
    bbox = bbox.transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(os.path.join(dir_name, filename), dpi="figure", bbox_inches=bbox)


def make_replacements(s, replacements):
    """ Make a bunch of replacements in a string """
    if s is None:
        return s

    for before, after in replacements:
        s = s.replace(before, after)

    return s


def pretty_dataset_name(dataset_name):
    """ Make dataset name look good for plots """
    return make_replacements(dataset_name, dataset_replacements)


def average_over_n(results, error_bars_over_runs_not_users=False):
    """ Average over multiple runs (values of n, the number of source domains)

    - Recompute mean/stdev for those that have multiple entries
    - Get rid of the n-specific dictionary

    i.e. we go from:
        results[dataset_name][method][n] = [
            (n, mean, std), ...
        ]
    to
        averaged_results[dataset_name][method] = [
            (n, mean, std), ...
        ]
    """
    # averaged_results[dataset_name][method] = []
    averaged_results = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )

    for dataset_name, v1 in results.items():
        for method_name, v2 in v1.items():
            new_values = []

            for n, values in v2.items():
                # Average over the multiple values here and recompute
                # the standard deviation
                if len(values) > 1:
                    values = np.array(values, dtype=np.float32)
                    if error_bars_over_runs_not_users:
                        # Average the errors of the per-user standard deviation
                        # over the multiple runs. This is instead of computing
                        # standard deviation over all runs with both varying
                        # target domain (user) and multiple runs -- which has
                        # an "error" conflating two separate aspects.
                        #
                        # Note: this applies only if we're averaging over users...
                        new_values.append((values[0, 0], values[:, 1].mean(),
                            values[:, 2].mean()))
                    else:
                        # All the 0th elements should be the same n
                        # Then recompute the mean/stdev from the accuracy values
                        # in 1th column
                        new_values.append((values[0, 0], values[:, 1].mean(),
                            values[:, 1].std(ddof=0)))
                elif len(values) == 1:
                    # Leave as is if there's only one
                    values = np.array(values, dtype=np.float32)
                    new_values.append((values[0, 0], values[0, 1],
                        values[0, 2]))
                else:
                    raise NotImplementedError("must be several or one run")

            # Sort on n
            new_values.sort(key=lambda x: x[0])

            averaged_results[dataset_name][method_name] = \
                np.array(new_values, dtype=np.float32)

    return averaged_results


def process_results(results, average_over_users, ssda, upper_bound_offset,
        tune, average_over_runs_per_user=True):
    """ Get results - get the test mean/std results indexed by:

        if not average, not ssda (i.e. msda):
            results[dataset_name + " " + target][method]
        if not average, ssda:
            results[(dataset_name, source(s), target)][method]
        if average, not ssda (i.e. msda):
            results[dataset_name][method]
        if average, ssda:
            results[dataset_name][method]

    Note: for example, dataset_name="ucihar", sources="1", target="2", and
    method="dann".
    """
    # results[dataset_name][method][n] = []
    # Note: at the end we average over the "n" dictionary
    processed_results = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
    )

    for result in results:
        method_name = result["method"]
        dataset_name = result["dataset"]
        dataset_name = pretty_dataset_name(dataset_name)

        # For single-source domain adaptation, we create a table for each
        # source -> target pair, so we need index by that.
        if (ssda and not average_over_users) or (not average_over_users and not average_over_runs_per_user):
            dataset_name = (dataset_name, result["sources"], result["target"])
        else:
            if method_name == "upper":
                dataset_name = (dataset_name, result["sources"])
            else:
                dataset_name = (dataset_name, result["target"])

        # What we want the x axis to be...
        if FLAGS.xaxis == "weight":
            n = result["similarity_weight"]
        elif FLAGS.xaxis == "uid":
            if upper_bound_offset is not None and result["method"] == "upper":
                n = result["uid"] + upper_bound_offset
            else:
                n = result["uid"]
        elif FLAGS.xaxis == "domains":
            n = len(result["sources"].split(","))  # number of source domains
        else:
            raise NotImplementedError("xaxis value needs to be weight or uid")

        # We care about the target domain (note for the upper bound, we
        # replaced the "target" value with "source" in _all_stats())
        mean, std, all_values = result["results_target_test"]

        processed_results[dataset_name][method_name][n].append(
            (n, mean, std))

        # Keep sorted by n
        processed_results[dataset_name][method_name][n].sort(key=lambda x: x[0])

    # Get rid of the n dictionary and average over the multiple values (should
    # only be >1 if average_over_users==True)
    processed_results = average_over_n(processed_results)

    # How we compute error bars -- only applies if we're averaging over users
    if average_over_users and FLAGS.error_bars_over_runs_not_users:
        # Currently it's indexed by: results[(dataset_name, user)][method]
        # Now we want to average over users for each dataset to get:
        # results[dataset_name][method]
        #
        # We can easily do this (and reuse existing code) by converting to
        # results[dataset_name][method][n] = [
        #     (n,mean,std) for user 1,
        #     (n,mean,std) for user 2, etc.
        # ] and averaging over the users with average_over_n, but now setting
        # error_bars_over_runs_not_users=True so we average the error over users
        # rather than recomputing overall all users/runs.
        new_processed_results = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(list)
            )
        )

        # Convert to results[(dataset_name,n)][method][user]
        for (dataset_name, user), values1 in processed_results.items():
            for method, values2 in values1.items():
                # Actually, values2 is a numpy array, so we need to save
                # [values3] to make sure average_over_n has a 2d array (not 1d)
                for values3 in values2:
                    n = values3[0]
                    # we want all the user data in one array so that
                    # average_over_n will average over that
                    new_processed_results[dataset_name][method][n].append(values3)

        # Average over users
        processed_results = average_over_n(new_processed_results,
            error_bars_over_runs_not_users=True)

    return processed_results


def dictionary_sorted_keys(d):
    keys = list(d.keys())
    keys.sort()
    return keys


def generate_plots(results, prefixes, save_plot=True, show_title=False,
        legend_separate=True, suffix="pdf", dir_name="result_plots",
        error_bars=True, figsize=(5, 3), skip=[], yrange=None,
        integer_axis=False, ncol=1, jitter_amount=0.01,
        x_is_percentage=False, y_is_percentage=True, title_suffix=""):
    # See: https://matplotlib.org/3.1.1/api/markers_api.html
    markers = ["o", "v", "^", "<", ">", "s", "p", "*", "D", "P", "X", "h",
       "1", "2", "3", "4", "+", "x", "d", "H", "|", "_"] * 2
    hollow = [False] * len(markers)

    # e.g. if "baselines" and "modats1", then "baselines,modats1" will be the
    # prefix
    prefix = ",".join(prefixes)

    # Do this sorted by name for a consistent ordering
    for dataset_name in dictionary_sorted_keys(results):
        dataset_values = results[dataset_name]
        methods = dictionary_sorted_keys(dataset_values)

        # Get data in order of the sorted methods
        data = [dataset_values[m] for m in methods]

        # Find min/max x values for scaling the jittering appropriately
        max_x = -np.inf
        min_x = np.inf
        for i in range(len(data)):
            method_data = np.array(data[i])
            x = method_data[:, 0]
            max_x = max(max(x), max_x)
            min_x = min(min(x), min_x)
        x_range = max_x - min_x

        # "dodge" points so they don't overlap
        jitter = gen_jitter(len(data), amount=jitter_amount*x_range)

        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=100)

        if yrange is not None:
            ax.set_ylim(yrange)

        # Only integers on x axis
        # https://stackoverflow.com/a/38096332
        if integer_axis:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        for i in range(len(data)):
            method_data = np.array(data[i])
            x = method_data[:, 0] + jitter[i]
            y = method_data[:, 1]
            std = method_data[:, 2]

            if x_is_percentage:
                x *= 100
            if y_is_percentage:
                y *= 100
                std *= 100

            if methods[i] in skip:
                continue

            if methods[i] in nice_method_names:
                method_name = nice_method_names[methods[i]]
            else:
                method_name = methods[i]

            if methods[i] in method_lines:
                line_type = method_lines[methods[i]]
            else:
                line_type = "-"

            if hollow[i]:
                mfc = "None"
            else:
                mfc = None

            if error_bars:
                p = plt.errorbar(x, y, yerr=std, label=method_name,
                    fmt=markers[i]+line_type, alpha=0.8, markerfacecolor=mfc)
            else:
                p = plt.plot(x, y, markers[i]+line_type, label=method_name,
                    alpha=0.8, markerfacecolor=mfc)

            # Make a horizontal line at the upper bound since it doesn't matter
            # what "n" is for this method (ignores the sources, only trains
            # on target)
            if methods[i] == "upper" and FLAGS.xaxis != "uid":
                # xmin=1 since the upper bound is 1 source in a sense
                assert method_lines[methods[i]] == "-.", \
                    "change linestyles in hlines to match that of method_lines[\"upper\"]"
                ax.hlines(y=y, xmin=1, xmax=max_x, colors=p[0].get_color(),
                    linestyles="dashdot")

        if show_title:
            plt.title("Dataset: " + dataset_name + title_suffix)

        if FLAGS.xaxis == "domains":
            ax.set_xlabel("Number of source domains")
        else:
            ax.set_xlabel(FLAGS.xaxis)

        ax.set_ylabel("Target Domain " + nice_metric_names[FLAGS.metric])

        if legend_separate:
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            legend = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=ncol)
            export_legend(legend, dir_name, filename=prefix+"_key."+suffix)
            legend.remove()
        else:
            # Put legend outside the graph http://stackoverflow.com/a/4701285
            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
            plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=ncol)

        if save_plot:
            save_dataset_name = dataset_name.replace(" ", "_")
            filename = prefix + "_" + save_dataset_name + "_" \
                + FLAGS.metric + "."+suffix
            plt.savefig(os.path.join(dir_name, filename),
                bbox_inches='tight')
            plt.close()

    if not save_plot:
        plt.show()


def get_float(value):
    """ Get float mean value (the part before plus/minus) from DDD.D $\pm$ DDD.D """
    float_value = None

    if len(value) > 0:
        parts = value.split(" $\pm$ ")

        if len(parts) == 1 or len(parts) == 2:
            if "underline{" in parts[0]:
                parts[0] = parts[0].replace("\\underline{", "")
                parts[1] = parts[1].replace("?", "")
            elif "textbf{" in parts[0]:
                parts[0] = parts[0].replace("\\textbf{", "")
                parts[1] = parts[1].replace("?", "")

            float_value = float(parts[0])

    return float_value


def replace_highest_better(values, references, better_text="textbf"):
    """ Replace DDD.D $\pm$ DDD.D with \textbf{...} if higher than some column(s) """
    # Get reference values first
    reference_values = []

    for i, v in enumerate(references):
        reference_values.append(get_float(v))

    # Modify if better than (or equal to) all reference values
    if len(reference_values) > 0:
        new_values = []

        for i, v in enumerate(values):
            float_value = get_float(v)

            if float_value is not None and float_value >= max(reference_values):
                new_values.append("\\" + better_text + "{"+v+"}")
            else:
                new_values.append(v)

        return new_values
    else:
        return values


def replace_highest_best(values, best_text="textbf"):
    """ Replace highest DDD.D $\pm$ DDD.D with \textbf{...} """
    max_index = []
    max_value = None

    for i, v in enumerate(values):
        float_value = get_float(v)

        if float_value is not None:
            if max_value is None or float_value > max_value:
                max_value = float_value
                max_index = [i]
            elif float_value == max_value:
                max_index.append(i)

    if max_index is not None:
        new_values = []

        for i, v in enumerate(values):
            if i in max_index:
                new_values.append("\\" + best_text + "{"+v+"}")
            else:
                new_values.append(v)

        return new_values
    else:
        return values


def write_table(output_filename, table, replace_best=None, best_bold=False,
        replace_better=None, replace_better_ref=None, better_bold=False):
    """
    Write Latex table to file,
    - underline highest row if replace_best=(row_start, row_end) inclusive
    - bold rows replace_better if better than all in replace_better_ref
    """
    best_command = "textbf" if best_bold else "underline"
    better_command = "textbf" if better_bold else "underline"

    with open(output_filename, "w") as f:
        max_columns = max([len(row) for row in table])
        f.write("\\begin{tabular}{" + "c"*max_columns + "}\n")

        for row in table:
            # \hline's
            if len(row) == 1:
                f.write(row[0]+"\n")
                continue

            # Identify best between columns if desired
            if replace_best is not None:
                try:
                    row_start, row_end = replace_best
                    row[row_start:row_end+1] = replace_highest_best(
                        row[row_start:row_end+1], best_command)
                except ValueError:
                    # If it's the header... ignore the error
                    pass

            if replace_better is not None:
                try:
                    row_start, row_end = replace_better
                    row_start_ref, row_end_ref = replace_better_ref
                    references = row[row_start_ref:row_end_ref+1]
                    row[row_start:row_end+1] = replace_highest_better(
                        row[row_start:row_end+1], references, better_command)
                except ValueError:
                    # If it's the header... ignore the error
                    pass

            for i, column in enumerate(row):
                f.write(column+" ")

                if i == len(row)-1:
                    f.write("\\\\\n")
                else:
                    f.write("& ")

        f.write("\\end{tabular}\n")


def generate_table(results, prefixes, output_filename, x_is_percentage=False,
        y_is_percentage=True, skip=[], list_of_methods=None, list_of_datasets=None,
        only_average=False, best_bold=False, better_bold=True,
        skip_best=False, skip_better=False, average=True, average_datasets=False):
    # indexed[dataset_name][n][method] = ""
    indexed = collections.defaultdict(
        lambda: collections.defaultdict(
            lambda: collections.defaultdict(str)
        )
    )
    # indexed_averaged_over_n[dataset_name][method] = []
    # Note: exclude Train on Target since it's one value for all n -- would be
    # the same averaged
    indexed_averaged_over_n = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    # Since Train on Target is one per dataset, do this separately
    # indexed_train_on_target[dataset_name] = ""
    indexed_train_on_target = collections.defaultdict(str)
    # indexed_average_over_datasets[method] = []
    indexed_average_over_datasets = collections.defaultdict(list)

    # If list of datasets isn't provided, then get the list of all of them,
    # sorted for consistency
    if list_of_datasets is None:
        list_of_datasets = dictionary_sorted_keys(results)

    for dataset_name in list_of_datasets:
        dataset_values = results[dataset_name]
        dataset_name = pretty_dataset_name(dataset_name)
        methods = dictionary_sorted_keys(dataset_values)

        # Get data in order of the sorted methods
        data = [dataset_values[m] for m in methods]

        for i in range(len(data)):
            method_data = np.array(data[i])
            x = method_data[:, 0]
            y = method_data[:, 1]
            std = method_data[:, 2]

            if x_is_percentage:
                x *= 100
            if y_is_percentage:
                y *= 100
                std *= 100

            if methods[i] in skip:
                continue

            if methods[i] in nice_method_names:
                method_name = nice_method_names[methods[i]]
            else:
                method_name = methods[i]

            if method_name == "Train on Target":
                for j, n in enumerate(x):
                    assert j == 0 and n == 1, \
                        "should only be one Train on Target"
                    val = "{:.1f} $\\pm$ {:.1f}".format(y[j], std[j])
                    indexed_train_on_target[dataset_name] = val

                    indexed_average_over_datasets[method_name].append([y[j], std[j]])
            else:
                for j, n in enumerate(x):
                    val = "{:.1f} $\\pm$ {:.1f}".format(y[j], std[j])
                    indexed[dataset_name][str(int(n))][method_name] = val

                    indexed_averaged_over_n[dataset_name][method_name].append(
                        [y[j], std[j]])

                    indexed_average_over_datasets[method_name].append([y[j], std[j]])

    #
    # Create Latex table
    #
    if list_of_methods is None:
        columns = ["No Adaptation", "CoDATS", "CALDA-Any,R", "CALDA-XS,H", "Train on Target"]
    else:
        columns = list_of_methods

    prepend_columns = ["Dataset"]

    if not only_average:
        prepend_columns += ["$n$"]

    fancy_columns = ["\\textit{"+c+"}" if "CALDA" in c else c for c in columns]

    # Create table
    table = []
    table.append(["\\toprule"])
    table.append(prepend_columns + fancy_columns)
    table.append(["\\midrule"])

    # Which datasets we want to include
    dataset_names = [
        dataset_name for dataset_name in indexed.keys()
        if (list_of_datasets is None or dataset_name in list_of_datasets)
    ]

    for i, dataset_name in enumerate(dataset_names):
        values1 = indexed[dataset_name]

        if not only_average:
            # Row for each value of n
            for n, values2 in values1.items():
                thisrow = [dataset_name, n]

                for method_name in columns:
                    if method_name == "Train on Target":
                        thisrow.append(indexed_train_on_target[dataset_name])
                    else:
                        if method_name in values2:
                            thisrow.append(values2[method_name])
                        else:
                            thisrow.append("")

                table.append(thisrow)

        if average and not only_average:
            # Horizontal dashed line, avg of each method for this dataset,
            # horizontal solid line
            table.append(["\\hdashline[0.75pt/3pt]"])  # https://tex.stackexchange.com/a/20141

        if average:
            thisrow = [dataset_name]

            if not only_average:
                thisrow += ["Avg"]

            for method_name in columns:
                # Train on Target is the same for each value of n, so averaged it
                # is also the same
                if method_name == "Train on Target":
                    thisrow.append(indexed_train_on_target[dataset_name])
                else:
                    if dataset_name in indexed_averaged_over_n \
                            and method_name in indexed_averaged_over_n[dataset_name]:
                        # Compute average
                        val = indexed_averaged_over_n[dataset_name][method_name]
                        val = np.array(val, dtype=np.float32)
                        # Format the same way and add to row
                        mean = val[:, 0].mean()
                        std = val[:, 1].mean()
                        val = "{:.1f} $\\pm$ {:.1f}".format(mean, std)
                        thisrow.append(val)
                    else:
                        thisrow.append("")

            table.append(thisrow)

        # Don't output for the last row, otherwise we have an extra line
        #
        # Also don't write if only-average, since then we have an hline for
        # every row.
        if i != len(dataset_names) - 1 and not only_average:
            table.append(["\\hline"])

    if average_datasets:
        table.append(["\\hline"])

        thisrow = ["Average"]

        if not only_average:
            thisrow += [""]  # don't have duplicate name

        for method_name in columns:
            if method_name in indexed_average_over_datasets:
                # Compute average
                val = indexed_average_over_datasets[method_name]
                val = np.array(val, dtype=np.float32)
                # Format the same way and add to row
                val = "{:.1f} $\\pm$ {:.1f}".format(val[:, 0].mean(), val[:, 1].mean())
                thisrow.append(val)
            else:
                thisrow.append("")

        table.append(thisrow)

    table.append(["\\bottomrule"])

    if skip_best:
        replace_best = None
    else:
        # Print table, but underline the highest in each row excluding prepended
        # columns
        replace_best_start = len(prepend_columns)
        replace_best_end = len(prepend_columns) + len(columns) - 1

        # Exclude train on target, if it's the last column (i.e., not doing the
        # ablation tables)
        if columns[-1] == "Train on Target":
            replace_best_end -= 1

        replace_best = (replace_best_start, replace_best_end)

    # Bold if better than CoDATS and No Adaptation
    if not skip_better and columns[0] == "No Adaptation" and "CoDATS" in columns[1]:
        # Reference is No Adaptation and CoDATS
        replace_better_ref_start = len(prepend_columns)  # No Adaptation
        replace_better_ref_end = len(prepend_columns)+1  # CoDATS
        replace_better_ref = (replace_better_ref_start, replace_better_ref_end)
        # End right before Train on Target (same as replace_best_end), but start
        # after CoDATS
        replace_better = (replace_better_ref_end+1, replace_best_end)
    elif not skip_better and columns[0] == "No Adaptation" and "CAN" in columns[1] and "CoDATS" in columns[2]:
        # Same as above but +2 for CoDATS since now we have CAN as well
        replace_better_ref_start = len(prepend_columns)  # No Adaptation
        replace_better_ref_end = len(prepend_columns)+2  # CoDATS - only difference
        replace_better_ref = (replace_better_ref_start, replace_better_ref_end)
        replace_better = (replace_better_ref_end+1, replace_best_end)
    else:
        replace_better = None
        replace_better_ref = None

    write_table(output_filename+".tex", table, replace_best=replace_best,
        replace_better=replace_better, replace_better_ref=replace_better_ref,
        best_bold=best_bold, better_bold=better_bold)


def compute_significance(results_averaged, results_not_averaged_over_targets,
        results_not_averaged_over_targets_or_runs, prefixes,
        list_of_methods=None, list_of_datasets=None,
        x_is_percentage=False, y_is_percentage=True,
        which="no_tr", limit_n_for_datasets=None):
    """
    Determine if there's a significant difference between two methods, i.e.
    len(list_of_methods) == 2

    First part where we get all values is based on generate_table() code
    """
    assert len(list_of_methods) == 2, \
        "significance() assumes comparison of two methods"

    if which == "no_t":
        results = results_not_averaged_over_targets
    elif which == "no_tr":
        results = results_not_averaged_over_targets_or_runs
    else:
        results = results_averaged

    # values[method] = []
    values = collections.defaultdict(list)

    for key, dataset_values in results.items():
        # Keys are different depending on if averaged or not
        if which == "no_t":
            dataset_name, target = key
        elif which == "no_tr":
            dataset_name, sources, target = key
        else:
            dataset_name = key

        dataset_name = pretty_dataset_name(dataset_name)
        methods = dictionary_sorted_keys(dataset_values)

        # Skip datasets we're not interested in
        if list_of_datasets is not None and dataset_name not in list_of_datasets:
            continue

        # Get data in order of the sorted methods
        data = [dataset_values[m] for m in methods]

        for i in range(len(data)):
            method_data = np.array(data[i])
            x = method_data[:, 0]
            y = method_data[:, 1]
            std = method_data[:, 2]

            if x_is_percentage:
                x *= 100
            if y_is_percentage:
                y *= 100
                std *= 100

            if methods[i] in nice_method_names:
                method_name = nice_method_names[methods[i]]
            else:
                method_name = methods[i]

            if method_name in list_of_methods:
                for j, n in enumerate(x):
                    assert n == int(x[j]), "x should be n but isn't"

                    if limit_n_for_datasets is not None:
                        # All synthetic datasets have same values of n
                        if "Synth " in dataset_name:
                            n_dataset_name = "synthetic"
                        else:
                            n_dataset_name = dataset_name

                        assert n_dataset_name in limit_n_for_datasets, \
                            "dataset " + n_dataset_name \
                            + " not in limit_n_for_datasets"

                        if n not in limit_n_for_datasets[n_dataset_name]:
                            # Skip this value since it's not the right
                            # value of n
                            continue

                    if which == "no_t":
                        name = "{} t={} n={}".format(dataset_name, target, n)
                    elif which == "no_tr":
                        name = "{} t={} s={} n={}".format(dataset_name, target, sources, n)
                    else:
                        name = "{} n={}".format(dataset_name, n)
                    values[method_name].append((name, y[j], std[j]))

    method1 = list_of_methods[0]
    method2 = list_of_methods[1]

    # Make sure "paired" values truly are paired
    method1_names = [x[0] for x in values[method1]]
    method2_names = [x[0] for x in values[method2]]
    assert len(method1_names) == len(method2_names), \
        "not the same number of values from each method"
    assert method1_names == method2_names, \
        "methods do not correspond, so paired test will not work"

    # Compute test
    method1_values = [x[1] for x in values[method1]]
    method2_values = [x[1] for x in values[method2]]

    test_notequal = stats.ttest_rel(method1_values, method2_values, alternative="two-sided").pvalue
    test_lessthan = stats.ttest_rel(method1_values, method2_values, alternative="less").pvalue

    return test_notequal, test_lessthan


def compute_significance_all(*args, list_of_methods, **kwargs):
    """ Compare how averaging affects significance tests """
    # How much (if any) averaging to use -- compare them all
    p_avg = compute_significance(*args, list_of_methods=list_of_methods, **kwargs, which="avg")[0]
    p_no_t = compute_significance(*args, list_of_methods=list_of_methods, **kwargs, which="no_t")[0]
    p_no_tr = compute_significance(*args, list_of_methods=list_of_methods, **kwargs, which="no_tr")[0]

    method1 = list_of_methods[0]
    method2 = list_of_methods[1]

    print("p-value {} != {}: avg {}, no_t {}, no_tr {}".format(
        method1, method2, p_avg, p_no_t, p_no_tr))

    # Return the one with no averaging
    return p_no_tr


def compute_significance_eqlt(*args, list_of_methods, **kwargs):
    """ Compare != vs. < significance tests """
    p_neq, p_lt = compute_significance(*args, list_of_methods=list_of_methods, **kwargs, which="no_tr")

    method1 = list_of_methods[0]
    method2 = list_of_methods[1]
    sig = ""

    if p_lt < 0.01:
        sig = ", significant 0.01"
    elif p_lt < 0.05:
        sig = ", significant 0.05"

    print("p-value {} != {}: {}, <: {}{}".format(
        method1, method2, p_neq, p_lt, sig))

    # Return the not equal one
    return p_neq


def make_plots_and_table(run_suffixes, variant_match, save_plot=True,
        show_title=False, legend_separate=True, ncol=4, suffix="pdf",
        skip=[], figsize=(5, 3), dir_name="result_plots",
        jitter_amount=0.005, source_feature_subset=None,
        target_feature_subset=None, upper_bound_offset=None, title_suffix="",
        tune=False, table_output_prefix=None, weak_supervision=False,
        domain_generalization=False, only_synthetic=False):
    """ Load files, process, save plots """
    results = get_results(run_suffixes, variant_match,
        source_feature_subset, target_feature_subset, tune)
    averages = process_results(results, average_over_users=True, ssda=False,
        upper_bound_offset=upper_bound_offset, tune=tune)
    not_averaged_over_targets = process_results(results, average_over_users=False, ssda=False,
        upper_bound_offset=upper_bound_offset, tune=tune)
    not_averaged_over_targets_or_runs = process_results(results, average_over_users=False, ssda=False,
        upper_bound_offset=upper_bound_offset, tune=tune, average_over_runs_per_user=False)

    generate_plots(averages, run_suffixes, save_plot,
        show_title, legend_separate, suffix, ncol=ncol, skip=skip,
        figsize=figsize, dir_name=dir_name, jitter_amount=jitter_amount,
        title_suffix=title_suffix)

    datasets_to_use = [
        "UCI HAR",
        "UCI HHAR",
        "WISDM AR",
        "WISDM AT",
        "Myo EMG",
        "NinaPro Myo",
    ]

    # Comparisons
    def _sig(m1, m2, list_of_datasets, **kwargs):
        compute_significance_eqlt(
            averages, not_averaged_over_targets, not_averaged_over_targets_or_runs, run_suffixes,
            list_of_methods=[m1, m2], list_of_datasets=list_of_datasets, **kwargs)

    def _sig_all(list_of_datasets, **kwargs):
        print("Datasets:", list_of_datasets)
        print("No Adversary - need adversary")
        _sig("CALDA-Any,R,NoAdv", "CALDA-Any,R", list_of_datasets, **kwargs)
        _sig("CALDA-XS,H,NoAdv", "CALDA-XS,H", list_of_datasets, **kwargs)
        print("Pseudo labeling - no pseudo labeling better")
        _sig("CALDA-In,R,P", "CALDA-In,R", list_of_datasets, **kwargs)
        _sig("CALDA-In,H,P", "CALDA-In,H", list_of_datasets, **kwargs)
        _sig("CALDA-Any,R,P", "CALDA-Any,R", list_of_datasets, **kwargs)
        _sig("CALDA-Any,H,P", "CALDA-Any,H", list_of_datasets, **kwargs)
        _sig("CALDA-XS,R,P", "CALDA-XS,R", list_of_datasets, **kwargs)
        _sig("CALDA-XS,H,P", "CALDA-XS,H", list_of_datasets, **kwargs)
        print("Selection - any/xs better than within")
        _sig("CALDA-In,R", "CALDA-Any,R", list_of_datasets, **kwargs)
        _sig("CALDA-In,R", "CALDA-XS,R", list_of_datasets, **kwargs)
        _sig("CALDA-In,H", "CALDA-Any,R", list_of_datasets, **kwargs)
        _sig("CALDA-In,H", "CALDA-XS,R", list_of_datasets, **kwargs)
        _sig("CALDA-In,R", "CALDA-Any,H", list_of_datasets, **kwargs)
        _sig("CALDA-In,R", "CALDA-XS,H", list_of_datasets, **kwargs)
        _sig("CALDA-In,H", "CALDA-Any,H", list_of_datasets, **kwargs)
        _sig("CALDA-In,H", "CALDA-XS,H", list_of_datasets, **kwargs)
        print("Selection - any better than xs")
        _sig("CALDA-XS,R", "CALDA-Any,H", list_of_datasets, **kwargs)
        _sig("CALDA-XS,H", "CALDA-Any,H", list_of_datasets, **kwargs)
        _sig("CALDA-XS,R", "CALDA-Any,R", list_of_datasets, **kwargs)
        _sig("CALDA-XS,H", "CALDA-Any,R", list_of_datasets, **kwargs)
        print("Hard selection - better to do hard sampling")
        _sig("CALDA-In,R", "CALDA-In,H", list_of_datasets, **kwargs)
        _sig("CALDA-Any,R", "CALDA-Any,H", list_of_datasets, **kwargs)
        _sig("CALDA-XS,R", "CALDA-XS,H", list_of_datasets, **kwargs)
        print("Hard selection - better to do random sampling")
        _sig("CALDA-In,H", "CALDA-In,R", list_of_datasets, **kwargs)
        _sig("CALDA-Any,H", "CALDA-Any,R", list_of_datasets, **kwargs)
        _sig("CALDA-XS,H", "CALDA-XS,R", list_of_datasets, **kwargs)
        print("Baselines")
        _sig("No Adaptation", "CALDA-Any,R", list_of_datasets, **kwargs)
        _sig("No Adaptation", "CALDA-XS,H", list_of_datasets, **kwargs)
        _sig("CoDATS", "CALDA-Any,R", list_of_datasets, **kwargs)
        _sig("CoDATS", "CALDA-XS,H", list_of_datasets, **kwargs)
        _sig("CAN", "CALDA-Any,R", list_of_datasets, **kwargs)
        _sig("CAN", "CALDA-XS,H", list_of_datasets, **kwargs)
        print()

    def _sig_all_dg(list_of_datasets, **kwargs):
        print("Datasets:", list_of_datasets)
        print("Domain Generalization - CALDG better")
        _sig("No Adaptation", "CALDG-Any,R", list_of_datasets, **kwargs)
        _sig("No Adaptation", "CALDG-XS,H", list_of_datasets, **kwargs)
        _sig("CoDATS-DG", "CALDG-Any,R", list_of_datasets, **kwargs)
        _sig("CoDATS-DG", "CALDG-XS,H", list_of_datasets, **kwargs)
        _sig("Sleep-DG", "CALDG-Any,R", list_of_datasets, **kwargs)
        _sig("Sleep-DG", "CALDG-XS,H", list_of_datasets, **kwargs)
        _sig("AFLAC-DG", "CALDG-Any,R", list_of_datasets, **kwargs)
        _sig("AFLAC-DG", "CALDG-XS,H", list_of_datasets, **kwargs)
        print("Domain Generalization - DA better than DG")
        _sig("CoDATS-DG", "CoDATS", list_of_datasets, **kwargs)
        _sig("CALDG-Any,R", "CALDA-Any,R", list_of_datasets, **kwargs)
        _sig("CALDG-XS,H", "CALDA-XS,H", list_of_datasets, **kwargs)
        print()

    def _sig_all_ws(list_of_datasets, **kwargs):
        print("Datasets:", list_of_datasets)
        print("Weak supervision - CALDA better")
        _sig("No Adaptation", "CALDA-Any,R,WS", list_of_datasets, **kwargs)
        _sig("No Adaptation", "CALDA-XS,H,WS", list_of_datasets, **kwargs)
        _sig("CoDATS-WS", "CALDA-Any,R,WS", list_of_datasets, **kwargs)
        _sig("CoDATS-WS", "CALDA-XS,H,WS", list_of_datasets, **kwargs)
        print("Weak supervision - WS better than no-WS")
        _sig("CALDA-Any,R", "CALDA-Any,R,WS", list_of_datasets, **kwargs)
        _sig("CALDA-XS,H", "CALDA-XS,H,WS", list_of_datasets, **kwargs)
        _sig("CoDATS", "CoDATS-WS", list_of_datasets, **kwargs)
        print()

    def _sig_just_n(list_of_datasets, f, which):
        if which == "low":
            limit_n_for_datasets = {
                "UCI HAR": [2],
                "UCI HHAR": [2],
                "WISDM AR": [2],
                "WISDM AT": [2],
                "Myo EMG": [2],
                "NinaPro Myo": [2],
                "synthetic": [2],
            }
        elif which == "low2":
            limit_n_for_datasets = {
                "UCI HAR": [2, 8],
                "UCI HHAR": [2, 3],
                "WISDM AR": [2, 8],
                "WISDM AT": [2, 12],
                "Myo EMG": [2, 10],
                "NinaPro Myo": [2, 4],
                "synthetic": [2, 4],
            }
        elif which == "high":
            limit_n_for_datasets = {
                "UCI HAR": [26],
                "UCI HHAR": [6],
                "WISDM AR": [26],
                "WISDM AT": [42],
                "Myo EMG": [34],
                "NinaPro Myo": [8],
                "synthetic": [10],
            }
        elif which == "high2":
            limit_n_for_datasets = {
                "UCI HAR": [20, 26],
                "UCI HHAR": [5, 6],
                "WISDM AR": [20, 26],
                "WISDM AT": [32, 42],
                "Myo EMG": [26, 34],
                "NinaPro Myo": [6, 8],
                "synthetic": [8, 10],
            }
        else:
            raise NotImplementedError("unknown which value passed")

        print("For", which, "values of n")
        f(list_of_datasets=list_of_datasets, limit_n_for_datasets=limit_n_for_datasets)

    def _generate_table(averages, run_suffixes, output_filename, *args, average=None, only_average=None, **kwargs):
        """ Generate both the normal table and the only-average table """
        # Normal table -- keep average as passed in
        generate_table(averages, run_suffixes, output_filename, *args, average=average, **kwargs)
        # Only-average table -- ignore average and instead pass only_average
        generate_table(averages, run_suffixes, output_filename+"_avg", *args, only_average=True, **kwargs)

    if table_output_prefix is not None:
        if not only_synthetic:
            if weak_supervision:
                _generate_table(averages, run_suffixes, table_output_prefix+"_ws",
                    skip=skip, list_of_methods=[
                        "No Adaptation",
                        "CoDATS-WS",
                        "CALDA-Any,R,WS",
                        "CALDA-XS,H,WS",
                        "Train on Target"
                    ], list_of_datasets=datasets_to_use, average=True, average_datasets=True)

                _sig_all_ws(datasets_to_use)

            if domain_generalization:
                _generate_table(averages, run_suffixes, table_output_prefix+"_dg",
                    skip=skip, list_of_methods=[
                        "No Adaptation",
                        "CoDATS-DG",
                        "Sleep-DG",
                        "AFLAC-DG",
                        "CALDG-Any,R",
                        "CALDG-XS,H",
                        "Train on Target"
                    ], list_of_datasets=datasets_to_use, average=True, average_datasets=True)

                _sig_all_dg(datasets_to_use)

            #
            # The full results
            #
            # Ablation
            _generate_table(averages, run_suffixes, table_output_prefix+"_ablation_selection",
                skip=skip, list_of_methods=[
                    # "CAN",
                    "CALDA-In,R",
                    "CALDA-In,H",
                    "CALDA-Any,R",
                    "CALDA-Any,H",
                    "CALDA-XS,R",
                    "CALDA-XS,H",
                ], list_of_datasets=datasets_to_use, average=True, best_bold=True, average_datasets=True)

            _generate_table(averages, run_suffixes, table_output_prefix+"_ablation_pseudo",
                skip=skip, list_of_methods=[
                    "CALDA-In,R,P",
                    "CALDA-In,H,P",
                    "CALDA-Any,R,P",
                    "CALDA-Any,H,P",
                    "CALDA-XS,R,P",
                    "CALDA-XS,H,P",
                ], list_of_datasets=datasets_to_use, average=True, skip_best=True, average_datasets=True)

            # No adversary ablation
            _generate_table(averages, run_suffixes, table_output_prefix+"_ablation_noadv",
                skip=skip, list_of_methods=[
                    "CALDA-Any,R,NoAdv",
                    "CALDA-XS,H,NoAdv",
                    "CALDA-Any,R",
                    "CALDA-XS,H",
                ], list_of_datasets=datasets_to_use, average=True, best_bold=True, average_datasets=True)

            _sig_all(datasets_to_use)

            # Full table
            _generate_table(averages, run_suffixes, table_output_prefix+"_full",
                skip=skip, list_of_methods=[
                    "No Adaptation",
                    "CAN",
                    "CoDATS",
                    "CALDA-Any,R",
                    "CALDA-XS,H",
                    "Train on Target"
                ], list_of_datasets=datasets_to_use, average=True, average_datasets=True)

        #
        # Synthetic data
        #
        synthetic_datasets = {
            "noshift": [
                "Synth InterT 0",
                "Synth InterR 0",
                "Synth IntraT 0",
                "Synth IntraR 0",
            ],
            "shift1": [
                "Synth InterT 5",
                "Synth InterR 0.5",
                "Synth IntraT 5",
                "Synth IntraR 0.5",
            ],
            "shift2": [
                "Synth InterT 10",
                "Synth InterR 1.0",
                "Synth IntraT 10",
                "Synth IntraR 1.0",
            ],
        }

        for name, datasets_to_use_synthetic in synthetic_datasets.items():
            _generate_table(averages, run_suffixes, table_output_prefix+"_ablation_selection_synthetic_"+name,
                skip=skip, list_of_methods=[
                    "CALDA-In,R",
                    "CALDA-In,H",
                    "CALDA-Any,R",
                    "CALDA-Any,H",
                    "CALDA-XS,R",
                    "CALDA-XS,H",
                ], list_of_datasets=datasets_to_use_synthetic, average=True, best_bold=True, average_datasets=True)

            _generate_table(averages, run_suffixes, table_output_prefix+"_ablation_pseudo_synthetic_"+name,
                skip=skip, list_of_methods=[
                    "CALDA-In,R,P",
                    "CALDA-In,H,P",
                    "CALDA-Any,R,P",
                    "CALDA-Any,H,P",
                    "CALDA-XS,R,P",
                    "CALDA-XS,H,P",
                ], list_of_datasets=datasets_to_use_synthetic, average=True, skip_best=True, average_datasets=True)

            _generate_table(averages, run_suffixes, table_output_prefix+"_full_synthetic_"+name,
                skip=skip, list_of_methods=[
                    "No Adaptation",
                    "CAN",
                    "CoDATS",
                    "CALDA-Any,R",
                    "CALDA-XS,H",
                    "Train on Target"
                ], list_of_datasets=datasets_to_use_synthetic, average=True, average_datasets=True),

            _generate_table(averages, run_suffixes, table_output_prefix+"_ablation_noadv_synthetic_"+name,
                skip=skip, list_of_methods=[
                    "CALDA-Any,R,NoAdv",
                    "CALDA-XS,H,NoAdv",
                    "CALDA-Any,R",
                    "CALDA-XS,H",
                ], list_of_datasets=datasets_to_use_synthetic, average=True, best_bold=True, average_datasets=True)

            _generate_table(averages, run_suffixes, table_output_prefix+"_ws_synthetic_"+name,
                    skip=skip, list_of_methods=[
                        "No Adaptation",
                        "CoDATS-WS",
                        "CALDA-Any,R,WS",
                        "CALDA-XS,H,WS",
                        "Train on Target"
                    ], list_of_datasets=datasets_to_use_synthetic, average=True, average_datasets=True)

            _generate_table(averages, run_suffixes, table_output_prefix+"_dg_synthetic_"+name,
                    skip=skip, list_of_methods=[
                        "No Adaptation",
                        "CoDATS-DG",
                        "Sleep-DG",
                        "AFLAC-DG",
                        "CALDG-Any,R",
                        "CALDG-XS,H",
                        "Train on Target"
                    ], list_of_datasets=datasets_to_use_synthetic, average=True, average_datasets=True)

            _sig_all(datasets_to_use_synthetic)
            _sig_all_dg(datasets_to_use_synthetic)
            _sig_all_ws(datasets_to_use_synthetic)


def main(argv):
    outdir = "result_plots"
    for_paper = FLAGS.paper
    skip = []

    if for_paper:
        outdir += "_paper"
        show_title = False
        legend_separate = True
        ncol = 5
        suffix = "pdf"
        figsize = (5, 3)
        jitter_amount = 0.005
    else:
        show_title = True
        legend_separate = False
        ncol = 1
        suffix = "png"
        figsize = (30, 18)
        jitter_amount = 0.005

    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # ACM doesn't like Type 3 fonts
    # https://tex.stackexchange.com/q/18687
    plt.rc('pdf', fonttype=42)
    plt.rc('ps', fonttype=42)

    # We pass variant_match as "*" rather than best_target since for the upper
    # bound there isn't a "target" (since target is passed as source), but we
    # all the others we evaluate only with best_target, so we can match all to
    # get the best_source only for the upper bound.
    def _make_plots_and_table(run_suffixes, upper_bound_offset=None,
            title_suffix="", tune=False, table_output_prefix=None,
            weak_supervision=False, domain_generalization=False,
            only_synthetic=False):
        make_plots_and_table(run_suffixes, "*",
            show_title=show_title, legend_separate=legend_separate, ncol=ncol,
            suffix=suffix, skip=skip, figsize=figsize, dir_name=outdir,
            jitter_amount=jitter_amount, upper_bound_offset=upper_bound_offset,
            title_suffix=title_suffix, tune=tune,
            table_output_prefix=table_output_prefix,
            weak_supervision=weak_supervision,
            domain_generalization=domain_generalization,
            only_synthetic=only_synthetic)

    _make_plots_and_table(["experiments"], table_output_prefix="results",
        weak_supervision=True, domain_generalization=True)


if __name__ == "__main__":
    app.run(main)

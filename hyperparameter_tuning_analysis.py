#!/usr/bin/env python3
"""
From a set of runs with varying parameters (i.e, a grid search), output which
was the best on average --> use those hyperparameters

(Simple and straightforward tuning method, which should be sufficient for our
limited number of hyperparameters)

Tuning options (see hyperparameter_tuning_experiments.py)

Instructions:

- Setup
  - Run ./hyperparameter_tuning_experiments.py > hyperparameter_tuning_experiments_list.py
  - Run ./experiments_msda.py --tune --name=tune
- Run
  - Upload scripts to Kamiak ./kamiak_upload.sh
  - Run tuning training script: sbatch kamiak_train_tune.srun tune
  - Run tuning eval script: sbatch kamiak_eval_tune.srun tune
- Analysis
  - Download the results with ./kamiak_download.sh
  - Find best hyperparameters with ./hyperparameter_tuning_analysis.py --prefixes=tune
- Use hyperparameters with ./experiments_msda.py --name=tune

Usage:
    ./hyperparameter_tuning_analysis.py --prefixes=tune > hyperparameters.py 2> >(tee hyperparameter_tuning_analysis.txt >&2)
"""
import sys
import collections

from absl import app
from absl import flags

from analysis import get_results, process_results, pretty_dataset_name
from print_dictionary import print_dictionary

# Generate with: ./hyperparameter_tuning_experiments.py > hyperparameter_tuning_experiments_list.py
from hyperparameter_tuning_experiments_list import hyperparameter_tuning_experiments_list
from hyperparameter_tuning_experiments_list_can import hyperparameter_tuning_experiments_list_can

FLAGS = flags.FLAGS

flags.DEFINE_string("prefixes", None, "The prefix(es) used during hyperparameter tuning, e.g. \"tune\" if the logs are in kamiak-logs-tune_lr0.00001_w100_p10_n20_t0.05, etc.")

# flags.mark_flag_as_required("prefixes")


def get_average_accuracy(folder, runs, dataset, method, average_over_users=True,
        debug=False, show_progress=False, only_n=None):
    # additional_match for speed
    results = get_results(runs, variant_match="*",
        source_feature_subset=None, target_feature_subset=None, tune=False,
        folder=folder, additional_match="{}-*-{}".format(dataset, method),
        show_progress=show_progress)
    averages = process_results(results, average_over_users=average_over_users,
        ssda=False, upper_bound_offset=None, tune=False)

    accuracies = []
    stdevs = []

    for dataset_name, dataset_values in averages.items():
        # Skip all data for datasets other than the one we care about
        if pretty_dataset_name(dataset) == dataset_name:
            for method_name, method_values in dataset_values.items():
                # Skip all the data for methods other than the one we care about
                if method == method_name:
                    for n, accuracy, stdev in method_values:
                        if debug:
                            print(dataset_name, method_name, n, accuracy, stdev,
                                file=sys.stderr)

                        if only_n is not None and n != only_n:
                            continue

                        accuracies.append(accuracy)
                        stdevs.append(stdev)

    if len(accuracies) > 0:
        return sum(accuracies) / len(accuracies), sum(stdevs) / len(stdevs)
    else:
        return -1, -1


def find_best_on_average(folder, prefixes, runs, dataset, method, debug=True):
    accuracies = [
        get_average_accuracy(folder, [
            prefix + "_" + run_name for prefix in prefixes
        ], dataset, method)
        for run_name, run_options, run_tuple in runs
    ]
    accuracies = [a for a, _ in accuracies]  # throw out stdev
    best_average_accuracy = max(accuracies)

    # Return the list of the best since it's possible more than one has the
    # same accuracy
    best_runs = []

    if best_average_accuracy != -1:
        if debug:
            print("    Average accuracy for each run:", file=sys.stderr)
            for i in range(len(runs)):
                # [0] is the folder, which is shorter than the full options in [1]
                print("      ", runs[i][0], accuracies[i], file=sys.stderr)
            print(file=sys.stderr)

        for i in range(len(runs)):
            if accuracies[i] == best_average_accuracy:
                best_runs.append(runs[i])

    return best_runs, best_average_accuracy


def main(argv):
    assert FLAGS.prefixes != "", "must pass prefixes"
    prefixes = FLAGS.prefixes.split(",")

    # Match that of hyperparameter_tuning_experiments.py probably
    datasets = [
        "ucihar", "ucihhar", "wisdm_ar", "wisdm_at",
        "myo", "ninapro_db5_like_myo_noshift",
        "normal_n12_l3_inter2_intra1_5,0,0,0_sine",
        "normal_n12_l3_inter2_intra1_0,0.5,0,0_sine",
        "normal_n12_l3_inter1_intra2_0,0,5,0_sine",
        "normal_n12_l3_inter1_intra2_0,0,0,0.5_sine",
    ]
    methods = ["none", "codats", "calda_xs_h", "upper", "can"]

    # hyperparameters...[dataset][method] = ""
    hyperparameters_str = collections.defaultdict(
        lambda: collections.defaultdict(str)
    )
    hyperparameters_tuple = collections.defaultdict(
        lambda: collections.defaultdict(str)
    )
    hyperparameters_folder = collections.defaultdict(
        lambda: collections.defaultdict(str)
    )

    for dataset in datasets:
        for method in methods:
            method_name_tuning = method

            if (
                (
                    dataset not in hyperparameter_tuning_experiments_list
                    or method_name_tuning not in hyperparameter_tuning_experiments_list[dataset]
                ) and (
                    dataset not in hyperparameter_tuning_experiments_list_can
                    or method_name_tuning not in hyperparameter_tuning_experiments_list_can[dataset]
                )
            ):
                print("Skipping", dataset, method, file=sys.stderr)
                continue

            print("Dataset:", dataset, file=sys.stderr)
            print("  Method:", method, file=sys.stderr)

            # Get runs for this method
            if method_name_tuning == "can":
                runs = hyperparameter_tuning_experiments_list_can[dataset][method_name_tuning]
            else:
                runs = hyperparameter_tuning_experiments_list[dataset][method_name_tuning]

            # Compute and output for debugging
            best_run, best_average_accuracy = find_best_on_average(
                "results_tune", prefixes, runs, dataset, method)
            print("    Use hyperparameters from (valid set):", ", ".join([
                run_folder for run_folder, run_options, run_tuple in best_run
            ]), file=sys.stderr)
            print("    which had average accuracy:", best_average_accuracy,
                file=sys.stderr)
            print(file=sys.stderr)
            print(file=sys.stderr)

            # Save hyperparameters
            if len(best_run) > 0:
                if len(best_run) > 1:
                    print("  Warning: more than one best run, selecting hyperparameters from the first",
                        file=sys.stderr)

                if best_average_accuracy != -1:
                    hyperparameters_str[dataset][method_name_tuning] = best_run[0][1]
                    hyperparameters_folder[dataset][method_name_tuning] = best_run[0][0]
                    hyperparameters_tuple[dataset][method_name_tuning] = best_run[0][2]
            else:
                print("  Warning: didn't find a best run",
                    file=sys.stderr)

    # Output hyperparameters (to stdout)
    print("""# Generated by hyperparameter_tuning_analysis.py. Changes will be overwritten.
def get(dataset, method, values, default_value=None):
    if dataset in values and method in values[dataset]:
        result = values[dataset][method]
    else:
        result = default_value

    return result


def get_hyperparameters_str(dataset, method):
    return get(dataset, method, hyperparameters_str, "")


def get_hyperparameters_tuple(dataset, method):
    return get(dataset, method, hyperparameters_tuple, None)


def get_hyperparameters_folder(dataset, method):
    return get(dataset, method, hyperparameters_folder, None)

""")

    print_dictionary(hyperparameters_str, "hyperparameters_str")
    print()
    print_dictionary(hyperparameters_tuple, "hyperparameters_tuple")
    print()
    print_dictionary(hyperparameters_folder, "hyperparameters_folder")


if __name__ == "__main__":
    app.run(main)

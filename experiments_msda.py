#!/usr/bin/env python3
"""
Generates the list of which multi-source adaptation problems to perform

For each dataset, for each target user, pick n random source users (excluding
the target user) 3 different times (so we can get mean +/- stdev).

Usage:
    ./experiments_msda.py --tune --name=tune
    ./experiments_msda.py --name=full

    ./experiments_msda.py --output_targets > experiments_targets.py
"""
import re
import os
import random
import collections

from absl import app
from absl import flags

import datasets.datasets as datasets

from print_dictionary import print_dictionary

# Generate with: ./hyperparameter_tuning_experiments.py > hyperparameter_tuning_experiments_list.py
# Used when passing --tune
from hyperparameter_tuning_experiments_list import hyperparameter_tuning_experiments_list
from hyperparameter_tuning_experiments_list_can import hyperparameter_tuning_experiments_list_can

# Generate with: ./hyperparameter_tuning_analysis.py > hyperparameters.py
# Used when not passing --tune
from hyperparameters import get_hyperparameters_str


FLAGS = flags.FLAGS

flags.DEFINE_boolean("tune", False, "Output scripts for hyperparameter tuning (use this before train scripts)")
flags.DEFINE_string("name", "", "Output to kamiak_{train,eval}_<output name>.srun")
flags.DEFINE_integer("train_cpu", 0, "Train with CPUs rather than GPUs for training script if train_cpu is > 0")
flags.DEFINE_integer("eval_cpu", 1, "CPUs rather than GPUs for evaluation script if eval_cpu is > 0")
flags.DEFINE_boolean("best_source", False, "Take best model on source validation set rather than target validation set")
flags.DEFINE_boolean("only_print", False, "Don't output train scripts, just print the list of experiments (--name not required)")
flags.DEFINE_boolean("output_targets", False, "Don't output train scripts, generate list of targets")
flags.DEFINE_boolean("can", False, "Output CAN hyperparameter tuning or experiments instead of CALDA")


# Get number of channels in the single-modality datasets
channels_for_dataset = {
    dataset: len(datasets.get_dataset(dataset).feature_names[0])
    for dataset in datasets.list_datasets_single_modality()
}
num_classes_for_dataset = {
    dataset: datasets.get_dataset(dataset).num_classes
    for dataset in datasets.list_datasets_single_modality()
}
baseline_methods = [
    "can",
]


def other_users(users, skip_user):
    """ From the list of users, throw out skip_user """
    new_users = []

    for user in users:
        if user != skip_user:
            new_users.append(user)

    return new_users


def generate_n_with_max(num_users, max_num, start_with=1):
    """ Generate [1,2,3,...,num_users] but max out at max_num and skip as close
    to evenly to get there. For example, if num_users=30 and max_num=5, we get:
    [1, 7, 13, 19, 25].

    Note: above example assumes start_with=1
    """
    return list(range(start_with, num_users, num_users//max_num))[:max_num]


def generate_multi_source(dataset_name, users, target_users, n, repeat, max_users):
    # Shrink the number of target users since otherwise we have >4000 adaptation
    # problems. That will take too long and won't fit in the paper's table
    # anyway.
    #
    # Take random set though, since IDs aren't necessarily randomized.
    # Note: not using random.shuffle() since that shuffles in-place
    shuffled_target_users = random.sample(target_users, len(target_users))
    possible_target_users = shuffled_target_users[:max_users]

    # We'll generate multi-source options for each target user
    pairs = []

    for target_user in possible_target_users:
        already_used_target = {}

        # We want several random subsets of each so we can get mean +/- stdev
        for i in range(repeat):
            skip = False

            # Select random source domains excluding target, keep shuffling until
            # we find a source set we haven't already used. The point of "repeat"
            # is to get *different* subsets. If it's the same, then there's not
            # much point in re-running with the exact same data.
            j = 0
            while True:
                others = other_users(users, target_user)
                random.shuffle(others)
                assert n <= len(others), "cannot choose n larger than len(users)-1"
                source_users = others[:n]

                # Sort so if we ever use the same subset, we don't have to
                # regenerate the files. Also easier to read.
                source_users.sort()

                if tuple(source_users) not in already_used_target:
                    already_used_target[tuple(source_users)] = None
                    break
                elif j > 1000:
                    print("Warning: couldn't pick different set of sources",
                        "than previously used,",
                        "dataset:"+dataset_name+",",
                        "n:"+str(n)+",",
                        "user:"+str(target_user)+",",
                        "repeat:"+str(i))
                    skip = True
                    break
                j += 1

            # Skip if this "repeat" would be the same as a previous one
            if skip:
                continue

            source_users = ",".join([str(x) for x in source_users])
            pairs.append((dataset_name, source_users, str(target_user)))

    return pairs


def generate_experiments_for_datasets(dataset_names, tuning):
    pairs = []
    uids = []

    for name in dataset_names:
        users = datasets.get_dataset_users(name)
        # Most of the time this is the same as users
        target_users = datasets.get_dataset_target_users(name)

        # Since sources-target aren't stored in filename anymore (too long), we
        # would run into folder name conflicts if we didn't append a unique ID
        # to each sources-target pair
        uid = 0

        # For each value of n, from 1 (single-source domain adaptation) up to
        # the full number of users - 1 (since we have one for the target)
        # options = generate_n_with_max(len(users), 5)

        # Now we need at least two domains, so start_with=2
        if tuning:
            options = generate_n_with_max(len(users), 2, start_with=2)
        else:
            options = generate_n_with_max(len(users), 5, start_with=2)

        for i, n in enumerate(options):
            # Make this repeatable even if we change which datasets, how many
            # n's we use, etc. Also nice since we end up using a subset of
            # n's source domains as (n-1)'s source domains. For example,
            # we get
            # (dataset_name, source_users, target_user) where each is a string
            # "sleep", "17", "0"
            # "sleep", "17,13", "0"
            # "sleep", "17,13,10", "0"
            # "sleep", "17,13,10,20", "0"
            random.seed(42)

            # Allows extra max_users for some datasets without changin uid's
            bonus_uid = 0

            if tuning:
                max_users = 5
                repeat = 3
            else:
                max_users = 10
                repeat = 3

            curr_pairs = generate_multi_source(name, users, target_users, n,
                repeat=repeat, max_users=max_users)

            for i, (dataset_name, source_users, target_user) in enumerate(curr_pairs):
                # We want to allow increasing the number of max_users for
                # wisdm_at and watch without changing the uid's of the 0-4
                # targets for backwards compatibility (otherwise we have to move
                # all the models around...)
                set_of_five = i // (5 * repeat)

                # before we had 0-4 (or 1-5), so do as before
                if max_users == 5 or set_of_five == 0:
                    uids.append(uid)
                    uid += 1
                else:
                    uids.append(str(uid)+"_"+str(bonus_uid))
                    bonus_uid += 1

            pairs += curr_pairs

    return pairs, uids


def print_experiments_list_debug(pairs, uids):
    print("List of adaptations we'll perform:")
    for i, (dataset_name, source, target) in enumerate(pairs):
        print("    ", dataset_name, source, "to", target, "uid", uids[i])
    print()


def output_list_of_targets(pairs, display=False):
    # list_of_targets[dataset_name] = [target1, target2, , ...]
    list_of_targets = collections.defaultdict(list)

    for i, (dataset_name, source, target) in enumerate(pairs):
        if target not in list_of_targets[dataset_name]:
            list_of_targets[dataset_name].append(target)

    if display:
        print_dictionary(list_of_targets, "list_of_targets")

    return list_of_targets


def atof(text):
    """ https://stackoverflow.com/a/5967539 """
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    """
    https://stackoverflow.com/a/5967539
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    """
    text = text[0] + text[1]  # we actually are sorting tuples of strings
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


def print_experiments_old(pairs, uids):
    # Train/eval baselines/weight
    print("For kamiak_{train,eval}...")
    dataset_names = []
    print_uids = []
    sources = []
    targets = []
    dataset_target_pairs = {}  # for upper bounds
    for i, (dataset_name, source, target) in enumerate(pairs):
        dataset_names.append("\""+dataset_name+"\"")
        print_uids.append(str(uids[i]))
        sources.append("\""+source+"\"")
        targets.append("\""+target+"\"")

        # for upper bounds
        pair_name = ("\""+dataset_name+"\"", "\""+target+"\"")
        full_pair = ("\""+dataset_name+"\"", str(uids[i]), "\""+target+"\"")
        if pair_name not in dataset_target_pairs:
            dataset_target_pairs[pair_name] = full_pair

    print("# number of adaptation problems =", len(sources))
    print("uids=(", " ".join(print_uids), ")", sep="")
    print("datasets=(", " ".join(dataset_names), ")", sep="")
    print("sources=(", " ".join(sources), ")", sep="")
    print("targets=(", " ".join(targets), ")", sep="")
    print()

    # Upper bound
    print("For kamiak_{train,eval}_upper")
    targets_unique = list(set(dataset_target_pairs.values()))
    targets_unique.sort(key=natural_keys)
    sources_blank = ["\"\""]*len(targets_unique)

    targets_unique_uids = []
    targets_unique_dataset = []
    targets_unique_target = []

    for dataset_name, uid, target in targets_unique:
        # Uses first uid from dataset_name-target
        targets_unique_uids.append(uid)
        targets_unique_dataset.append(dataset_name)
        targets_unique_target.append(target)

    print("# number of adaptation problems =", len(targets_unique))
    print("uids=(", " ".join(["u"+str(x) for x in targets_unique_uids]), ")", sep="")
    print("datasets=(", " ".join(targets_unique_dataset), ")", sep="")
    print("sources=(", " ".join(sources_blank), ")", sep="")
    print("targets=(", " ".join(targets_unique_target), ")", sep="")
    print()


def fill_in_template(filename, replacements):
    with open(filename, "r") as f:
        template = f.read()  # .decode("utf-8")?

    for name, replacement in replacements.items():
        template = template.replace("{{" + name + "}}", str(replacement))

    return template


def save_script(filename, contents):
    with open(filename, "w") as f:
        f.write(contents)


def upper_from_full(pairs, uids):
    dataset_target_pairs = {}
    for i, (dataset_name, source, target) in enumerate(pairs):
        pair_name = (dataset_name, target)
        full_pair = (dataset_name, str(uids[i]), target)
        if pair_name not in dataset_target_pairs:
            dataset_target_pairs[pair_name] = full_pair

    targets_unique = list(set(dataset_target_pairs.values()))
    targets_unique.sort(key=natural_keys)

    upper_uids = []
    upper_pairs = []

    for dataset_name, uid, target in targets_unique:
        source = ""
        upper_uids.append("u{}".format(uid))
        upper_pairs.append((
            dataset_name, source, target
        ))

    return upper_pairs, upper_uids


def list_to_str(values):
    return " ".join(values)


def get_cpu_or_gpu(cpu):
    if cpu == 0:
        cpu_or_gpu = "gpu"
        partitions = "cook,cahnrs_gpu,kamiak"  # for kamiak
        # partitions = "cook" # for my desktop
        gpus = 1
        cpus = 1
    else:
        cpu_or_gpu = "cpu"
        partitions = "cook,vcea,cahnrs,kamiak"
        gpus = 0
        cpus = cpu

    return cpu_or_gpu, partitions, gpus, cpus


def output_experiments(name, method_names,
        full_pairs, full_uids, upper_pairs, upper_uids, tuning,
        train_cpu, eval_cpu, eval_variant="best_target"):
    # Create experiments for each method. These are different for train/eval
    # only because we use multiple debugnums for the upper bound.
    methods_train = []
    debugnums_train = []
    uids_train = []
    datasets_train = []
    sources_train = []
    targets_train = []
    options_train = []
    additional_suffixes_train = []

    methods_eval = []
    variants_eval = []
    uids_eval = []
    datasets_eval = []
    sources_eval = []
    targets_eval = []
    additional_suffixes_eval = []

    # For Baselines
    baseline_can_problem_names = []
    baseline_can_source_names = []
    baseline_can_target_name = []
    baseline_can_test_name = []
    baseline_can_num_classes = []
    baseline_can_in_channels = []
    baseline_can_save_dir = []
    baseline_can_results_filenames = []
    baseline_can_datasets = []
    baseline_can_sources = []
    baseline_can_targets = []
    baseline_can_uids = []
    baseline_can_options = []

    # For each method
    for method_name in method_names:
        if method_name == "upper":
            method_pairs = upper_pairs
        else:
            method_pairs = full_pairs

        # For each adaptation problem/experiment, e.g. adapt sources (1,2)
        # to 3 (of dataset D and method M).
        for i, (dataset_name, source, target) in enumerate(method_pairs):
            # Whether this is a baseline or our method
            is_baseline = method_name in baseline_methods

            # If tuning, we iterate over possible hyperparameters
            if tuning:
                if is_baseline:
                    assert method_name == "can", \
                        "only CAN tuning supported for now"

                    if dataset_name in hyperparameter_tuning_experiments_list_can \
                            and method_name in hyperparameter_tuning_experiments_list_can[dataset_name]:
                        hyperparameter_set = hyperparameter_tuning_experiments_list_can[dataset_name][method_name]
                    else:
                        # Skip this method since we aren't tuning it
                        continue
                else:
                    if dataset_name in hyperparameter_tuning_experiments_list \
                            and method_name in hyperparameter_tuning_experiments_list[dataset_name]:
                        hyperparameter_set = hyperparameter_tuning_experiments_list[dataset_name][method_name]
                    else:
                        # Skip this method since we aren't tuning it
                        continue
            # Otherwise, just set to None so we go through the loop once (and never
            # use these values)
            else:
                hyperparameter_set = [(None, None, None)]
                additional_suffix = ""
                option = None

            # If tuning=False, then hyperparameter_set is just None's so we do this
            # loop once. Otherwise, we do it once per set of hyperparameters we want
            # to test.
            for hyperparam_folder, hyperparam_option, hyperparam_tuple in hyperparameter_set:
                # Initially, we don't know what hyperparameters to use. Thus, during
                # tuning we try a bunch of different ones and append a suffix based
                # on that set of parameters.
                if tuning:
                    additional_suffix = "_" + hyperparam_folder
                    option = hyperparam_option
                else:
                    additional_suffix = ""

                    # Hyperparameters if available -- sometimes use the parameters
                    # from a different method for direct comparison.
                    if "calda" in method_name or "caldg" in method_name:
                        hyp_method_name = "calda_xs_h"
                    elif "codats" in method_name or method_name in ["sleep_dg", "aflac_dg"]:
                        hyp_method_name = "codats"
                    else:
                        hyp_method_name = method_name

                    if dataset_name in ["normal_n12_l3_inter0_intra1_5,0,0,0_sine", "normal_n12_l3_inter1_intra1_5,0,0,0_sine", "normal_n12_l3_inter2_intra1_5,0,0,0_sine"]:
                        hyp_dataset_name = "normal_n12_l3_inter2_intra1_5,0,0,0_sine"
                    elif dataset_name in ["normal_n12_l3_inter0_intra1_0,0.5,0,0_sine", "normal_n12_l3_inter1_intra1_0,0.5,0,0_sine", "normal_n12_l3_inter2_intra1_0,0.5,0,0_sine"]:
                        hyp_dataset_name = "normal_n12_l3_inter2_intra1_0,0.5,0,0_sine"
                    elif dataset_name in ["normal_n12_l3_inter1_intra0_0,0,5,0_sine", "normal_n12_l3_inter1_intra1_0,0,5,0_sine", "normal_n12_l3_inter1_intra2_0,0,5,0_sine"]:
                        hyp_dataset_name = "normal_n12_l3_inter1_intra2_0,0,5,0_sine"
                    elif dataset_name in ["normal_n12_l3_inter1_intra0_0,0,0,0.5_sine", "normal_n12_l3_inter1_intra1_0,0,0,0.5_sine", "normal_n12_l3_inter1_intra2_0,0,0,0.5_sine"]:
                        hyp_dataset_name = "normal_n12_l3_inter1_intra2_0,0,0,0.5_sine"
                    elif "normal_n" in dataset_name:
                        raise NotImplementedError("Found unknown normal-synthetic dataset")
                    else:
                        hyp_dataset_name = dataset_name

                    option = get_hyperparameters_str(hyp_dataset_name, hyp_method_name)

                if method_name == "upper":
                    uid = upper_uids[i]

                    # Upper bound uses multiple debugnums for multiple runs rather
                    # than multiple sets of source domains (since there is no
                    # source domain)
                    debugnums = [1, 2, 3]
                else:
                    uid = full_uids[i]

                    # Only upper bound has more than one debugnum. The rest have the
                    # multiple runs through multiple different sets of sources.
                    debugnums = [1]

                if is_baseline:
                    if method_name == "can":
                        for debugnum in debugnums:
                            baseline_can_problem_names.append(
                                "can_timeseries_{dataset_name}_{uid}_{debugnum}{suffix}".format(
                                    dataset_name=dataset_name,
                                    uid=uid,
                                    debugnum=debugnum,
                                    suffix=additional_suffix,
                                )
                            )
                            baseline_can_results_filenames.append(
                                "results_{suffix}_best_target-{dataset_name}-{uid}-{method}".format(
                                    suffix=name+additional_suffix,
                                    dataset_name=dataset_name,
                                    uid=uid,
                                    method=method_name,
                                )
                            )
                            baseline_can_source_names.append([
                                "{dataset_name}_{domain}_train".format(
                                    dataset_name=dataset_name,
                                    domain=s,
                                )
                                for s in source.split(",")
                            ])
                            baseline_can_target_name.append(
                                "{dataset_name}_{domain}_train".format(
                                    dataset_name=dataset_name,
                                    domain=target,
                                )
                            )
                            baseline_can_test_name.append(
                                "{dataset_name}_{domain}_{test}".format(
                                    dataset_name=dataset_name,
                                    domain=target,
                                    # Don't look at real test set during tuning
                                    test="valid" if tuning else "test"
                                )
                            )
                            baseline_can_num_classes.append(
                                num_classes_for_dataset[dataset_name]
                            )
                            baseline_can_in_channels.append(
                                channels_for_dataset[dataset_name]
                            )
                            baseline_can_save_dir.append(
                                "./experiments/ckpt{debugnum}".format(
                                    debugnum=debugnum,
                                )
                            )
                            baseline_can_datasets.append("\"{}\"".format(dataset_name))
                            baseline_can_sources.append("\"{}\"".format(source))
                            baseline_can_targets.append("\"{}\"".format(target))
                            baseline_can_uids.append("\"{}\"".format(uid))
                            if option is not None:
                                baseline_can_options.append("\"{}\"".format(option))
                            else:
                                baseline_can_options.append("\"\"")
                    else:
                        raise NotImplementedError("baseline " + method_name)
                else:
                    for debugnum in debugnums:
                        # Train array values
                        methods_train.append("\"{}\"".format(method_name))
                        debugnums_train.append("\"{}\"".format(debugnum))
                        uids_train.append("\"{}\"".format(uid))
                        datasets_train.append("\"{}\"".format(dataset_name))
                        sources_train.append("\"{}\"".format(source))
                        targets_train.append("\"{}\"".format(target))
                        options_train.append("\"{}\"".format(option))
                        additional_suffixes_train.append("\"{}\"".format(additional_suffix))

                    # When doing the upper bound, we set the source to be the target and
                    # have no target. Thus, we need to select based on the best source
                    # instead.
                    if method_name == "upper":
                        variant = "best_source"
                    else:
                        variant = eval_variant

                    # Eval array values
                    methods_eval.append("\"{}\"".format(method_name))
                    variants_eval.append("\"{}\"".format(variant))
                    uids_eval.append("\"{}\"".format(uid))
                    datasets_eval.append("\"{}\"".format(dataset_name))
                    sources_eval.append("\"{}\"".format(source))
                    targets_eval.append("\"{}\"".format(target))
                    additional_suffixes_eval.append("\"{}\"".format(additional_suffix))

    # Sanity check (also checked in the .srun files)
    assert len(methods_train) == len(debugnums_train), "debugnums_train wrong length"
    assert len(methods_train) == len(uids_train), "uids_train wrong length"
    assert len(methods_train) == len(datasets_train), "datasets_train wrong length"
    assert len(methods_train) == len(sources_train), "sources_train wrong length"
    assert len(methods_train) == len(targets_train), "targets_train wrong length"
    assert len(methods_train) == len(options_train), "options_train wrong length"
    assert len(methods_train) == len(additional_suffixes_train), "additional_suffixes_train wrong length"
    assert len(methods_eval) == len(variants_eval), "variants_eval wrong length"
    assert len(methods_eval) == len(uids_eval), "uids_eval wrong length"
    assert len(methods_eval) == len(datasets_eval), "datasets_eval wrong length"
    assert len(methods_eval) == len(sources_eval), "sources_eval wrong length"
    assert len(methods_eval) == len(targets_eval), "targets_eval wrong length"
    assert len(methods_eval) == len(additional_suffixes_eval), "additional_suffixes_eval wrong length"

    # Fill the values into the templates
    cpu_or_gpu_train, partitions_train, gpus_train, cpus_train = get_cpu_or_gpu(train_cpu)
    cpu_or_gpu_eval, partitions_eval, gpus_eval, cpus_eval = get_cpu_or_gpu(eval_cpu)

    # If we have any of our method (i.e. not just baselines)
    if len(methods_train) > 0:
        if tuning:
            results_dir = "results_tune"
            # For hyperparameter tuning, we pass --notest so tuning only looks at
            # the validation set, never the real test set.
            additional_args = "--notest"
        else:
            results_dir = "results"
            additional_args = ""

        train_script = fill_in_template("kamiak_train.srun.template", {
            "cpus": cpus_train,
            "gpus": gpus_train,
            "partitions": partitions_train,
            "max_array": len(methods_train) - 1,
            "cpu_or_gpu": cpu_or_gpu_train,
            "methods": list_to_str(methods_train),
            "debugnums": list_to_str(debugnums_train),
            "uids": list_to_str(uids_train),
            "datasets": list_to_str(datasets_train),
            "sources": list_to_str(sources_train),
            "targets": list_to_str(targets_train),
            "options": list_to_str(options_train),
            "additional_suffixes": list_to_str(additional_suffixes_train),
        })

        eval_script = fill_in_template("kamiak_eval.srun.template", {
            "cpus": cpus_eval,
            "gpus": gpus_eval,
            "partitions": partitions_eval,
            "max_array": len(methods_eval) - 1,
            "cpu_or_gpu": cpu_or_gpu_eval,
            "results_dir": results_dir,
            "methods": list_to_str(methods_eval),
            "variants": list_to_str(variants_eval),  # variants instead of debugnums
            "uids": list_to_str(uids_eval),
            "datasets": list_to_str(datasets_eval),
            "sources": list_to_str(sources_eval),
            "targets": list_to_str(targets_eval),
            # no options
            "additional_suffixes": list_to_str(additional_suffixes_eval),
            "additional_args": additional_args,
        })

        # Save scripts
        train_script_filename = "kamiak_train_" + name + ".srun"
        eval_script_filename = "kamiak_eval_" + name + ".srun"
        print("Writing", train_script_filename, eval_script_filename)
        save_script(train_script_filename, train_script)
        save_script(eval_script_filename, eval_script)

    # If we have baselines
    num_can_baselines = len(baseline_can_source_names)
    if num_can_baselines > 0:
        # Generate config files for CAN
        for i in range(num_can_baselines):
            train_script = fill_in_template("can_train.yml.template", {
                "num_classes": baseline_can_num_classes[i],
                "sources_train": baseline_can_source_names[i],
                "target_train": baseline_can_target_name[i],
                "in_channels": baseline_can_in_channels[i],
                "target_test": baseline_can_test_name[i],
                "save_dir": baseline_can_save_dir[i],
            })

            eval_script = fill_in_template("can_eval.yml.template", {
                "num_classes": baseline_can_num_classes[i],
                "in_channels": baseline_can_in_channels[i],
                "target_test": baseline_can_test_name[i],
                "save_dir": baseline_can_save_dir[i],
            })

            # Save scripts
            config_name = baseline_can_problem_names[i]
            base = "../Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation/experiments/config/{name}".format(name=config_name)
            can_base = os.path.join(base, "CAN")
            train_script_filename = "{can_base}/{name}_train_train2val_cfg.yaml".format(can_base=can_base, name=config_name)
            eval_script_filename = "{base}/{name}_test_val_cfg.yaml".format(base=base, name=config_name)

            if not os.path.exists(base):
                os.makedirs(base)
            if not os.path.exists(can_base):
                os.makedirs(can_base)

            print("Writing", train_script_filename, eval_script_filename)
            save_script(train_script_filename, train_script)
            save_script(eval_script_filename, eval_script)

        train_script = fill_in_template("kamiak_baseline_can_train.srun.template", {
            "cpus": cpus_train,
            "gpus": gpus_train,
            "partitions": partitions_train,
            "max_array": len(baseline_can_problem_names) - 1,
            "cpu_or_gpu": cpu_or_gpu_train,
            "names": list_to_str(baseline_can_problem_names),
            "savedirs": list_to_str(baseline_can_save_dir),
            "options": list_to_str(baseline_can_options),
        })

        eval_script = fill_in_template("kamiak_baseline_can_eval.srun.template", {
            "cpus": cpus_eval,
            "gpus": gpus_eval,
            "partitions": partitions_eval,
            "max_array": len(baseline_can_problem_names) - 1,
            "cpu_or_gpu": cpu_or_gpu_eval,
            "names": list_to_str(baseline_can_problem_names),
            "savedirs": list_to_str(baseline_can_save_dir),
            "output_filenames": list_to_str(baseline_can_results_filenames),
            "datasets": list_to_str(baseline_can_datasets),
            "sources": list_to_str(baseline_can_sources),
            "targets": list_to_str(baseline_can_targets),
            "uids": list_to_str(baseline_can_uids),
            "out_dir": "results_tune" if tuning else "results",
        })

        # Save scripts
        train_script_filename = "kamiak_train_baseline_can_" + name + ".srun"
        eval_script_filename = "kamiak_eval_baseline_can_" + name + ".srun"
        print("Writing", train_script_filename, eval_script_filename)
        save_script(train_script_filename, train_script)
        save_script(eval_script_filename, eval_script)


def main(argv):
    tuning = FLAGS.tune

    #
    # All methods we wish to use
    #
    # For which method name in the paper corresponds to which method name here,
    # see analysis.py
    #
    method_names = []

    if FLAGS.can:
        method_names = ["can"]
    else:
        # Main results
        method_names += ["calda_xs_h", "calda_any_r", "codats", "none", "upper"]

        if not tuning:
            # Weak supervision
            method_names += ["codats_ws", "calda_xs_h_ws", "calda_any_r_ws"]
            # Domain generalization
            method_names += ["codats_dg", "caldg_xs_h", "caldg_any_r", "sleep_dg", "aflac_dg"]
            # CALDA - only contrastive, no adversary
            method_names += ["calda_xs_h_noadv", "calda_any_r_noadv"]

        method_names += ["calda_xs_r", "calda_in_r", "calda_xs_r_p", "calda_in_r_p", "calda_any_r_p", "calda_in_h", "calda_any_h", "calda_xs_h_p", "calda_in_h_p", "calda_any_h_p"]

    #
    # All datasets we wish to use
    #
    dataset_names = []
    # HAR datasets
    dataset_names += ["ucihar", "ucihhar", "wisdm_ar", "wisdm_at"]
    # EMG datasets
    dataset_names += ["myo", "ninapro_db5_like_myo_noshift"]
    # Synthetic datasets
    if tuning:
        dataset_names += [
            "normal_n12_l3_inter2_intra1_5,0,0,0_sine",
            "normal_n12_l3_inter2_intra1_0,0.5,0,0_sine",
            "normal_n12_l3_inter1_intra2_0,0,5,0_sine",
            "normal_n12_l3_inter1_intra2_0,0,0,0.5_sine",
        ]
    else:
        dataset_names += [
            "normal_n12_l3_inter0_intra1_5,0,0,0_sine",
            "normal_n12_l3_inter1_intra1_5,0,0,0_sine",
            "normal_n12_l3_inter2_intra1_5,0,0,0_sine",
            "normal_n12_l3_inter0_intra1_0,0.5,0,0_sine",
            "normal_n12_l3_inter1_intra1_0,0.5,0,0_sine",
            "normal_n12_l3_inter2_intra1_0,0.5,0,0_sine",
            "normal_n12_l3_inter1_intra0_0,0,5,0_sine",
            "normal_n12_l3_inter1_intra1_0,0,5,0_sine",
            "normal_n12_l3_inter1_intra2_0,0,5,0_sine",
            "normal_n12_l3_inter1_intra0_0,0,0,0.5_sine",
            "normal_n12_l3_inter1_intra1_0,0,0,0.5_sine",
            "normal_n12_l3_inter1_intra2_0,0,0,0.5_sine",
        ]

    # Generate full list of experiments
    full_pairs, full_uids = generate_experiments_for_datasets(dataset_names, tuning)
    upper_pairs, upper_uids = upper_from_full(full_pairs, full_uids)

    if FLAGS.output_targets:
        output_list_of_targets(full_pairs, display=True)
    else:
        # Check that these make sense by printing list
        print_experiments_list_debug(full_pairs, full_uids)

        if not FLAGS.only_print:
            assert FLAGS.name != "", "Need to pass argument --name=<name>"

            # Command line settings
            eval_variant = "best_source" if FLAGS.best_source else "best_target"

            # Create a train/eval script
            output_experiments(FLAGS.name, method_names,
                full_pairs, full_uids, upper_pairs, upper_uids, tuning,
                train_cpu=FLAGS.train_cpu, eval_cpu=FLAGS.eval_cpu,
                eval_variant=eval_variant)


if __name__ == "__main__":
    app.run(main)

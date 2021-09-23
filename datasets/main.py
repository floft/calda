#!/usr/bin/env python3
"""
Process each dataset into .tfrecord files

Run (or see ../generate_tfrecords.sh script):

    python -m datasets.main <args>

Note: probably want to run this prefixed with CUDA_VISIBLE_DEVICES= so that it
doesn't use the GPU (if you're running other jobs). Does this by default if
parallel=True since otherwise it'll error.
"""
import os
import sys
import numpy as np
import tensorflow as tf

from absl import app
from absl import flags
from sklearn.model_selection import train_test_split

from datasets import datasets
from pool import run_job_pool
from datasets.tfrecord import write_tfrecord, write_tfrecord_modality, \
    tfrecord_filename
from datasets.normalization import calc_normalization_modality, \
    apply_normalization_modality

FLAGS = flags.FLAGS

flags.DEFINE_boolean("parallel", True, "Run multiple in parallel")
flags.DEFINE_integer("jobs", 0, "Parallel jobs (if parallel=True), 0 = # of CPU cores")
flags.DEFINE_boolean("debug", False, "Whether to print debug information")


def write(filename, x, y):
    if x is not None and y is not None:
        if not os.path.exists(filename):
            write_tfrecord(filename, x, y)
        elif FLAGS.debug:
            print("Skipping:", filename, "(already exists)")
    elif FLAGS.debug:
        print("Skipping:", filename, "(no data)")


def write_modality(filename, xs, y):
    """ Same as write() except calls the multi-modality version """
    if xs is not None and y is not None:
        if not os.path.exists(filename):
            write_tfrecord_modality(filename, xs, y)
        elif FLAGS.debug:
            print("Skipping:", filename, "(already exists)")
    elif FLAGS.debug:
        print("Skipping:", filename, "(no data)")


def shuffle_together_calc(length, seed=None):
    """ Generate indices of numpy array shuffling, then do x[p] """
    rand = np.random.RandomState(seed)
    p = rand.permutation(length)
    return p


def to_numpy(value):
    """ Make sure value is numpy array """
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return value


def valid_split(data, labels, seed=None, validation_size=1000):
    """ (Stratified) split training data into train/valid as is commonly done,
    taking 1000 random (stratified) (labeled, even if target domain) samples for
    a validation set """
    percentage_size = int(0.2*len(data))
    if percentage_size > validation_size:
        test_size = validation_size
    else:
        if FLAGS.debug:
            print("Warning: using smaller validation set size", percentage_size)
        test_size = 0.2  # 20% maximum

    x_train, x_valid, y_train, y_valid = \
        train_test_split(data, labels, test_size=test_size,
            stratify=labels, random_state=seed)

    return x_train, y_train, x_valid, y_valid


def valid_split_modality(xs, y, seed=None, validation_size=1000):
    """ (Stratified) split training data into train/valid as is commonly done,
    taking 1000 random (stratified) (labeled, even if target domain) samples for
    a validation set """
    percentage_size = int(0.2*len(xs[0]))
    if percentage_size > validation_size:
        test_size = validation_size
    else:
        if FLAGS.debug:
            print("Warning: using smaller validation set size", percentage_size)
        test_size = 0.2  # 20% maximum

    # Returns train/test for each input passed in, so: xs1_train, xs1_test,
    # xs2_train, xs2_test, ...
    results = train_test_split(*xs, y, test_size=test_size,
        stratify=y, random_state=seed)

    # Train is evens (starting at position 0), test is odds
    assert len(results) % 2 == 0, "should get even number of splits"
    train = results[0::2]
    valid = results[1::2]

    # y is at the end, xs is everything else
    xs_train = train[:-1]
    xs_valid = valid[:-1]
    y_train = train[-1]
    y_valid = valid[-1]

    return xs_train, y_train, xs_valid, y_valid


def save_dataset(dataset_name, output_dir, seed=0):
    """ Save single dataset """
    train_filename = os.path.join(output_dir,
        tfrecord_filename(dataset_name, "train"))
    valid_filename = os.path.join(output_dir,
        tfrecord_filename(dataset_name, "valid"))
    test_filename = os.path.join(output_dir,
        tfrecord_filename(dataset_name, "test"))

    # Skip if they already exist
    if os.path.exists(train_filename) \
            and os.path.exists(valid_filename) \
            and os.path.exists(test_filename):
        if FLAGS.debug:
            print("Skipping:", train_filename, valid_filename, test_filename,
               "already exist")
        return

    if FLAGS.debug:
        print("Saving dataset", dataset_name)
        sys.stdout.flush()

    dataset, dataset_class = datasets.load(dataset_name)

    # Skip if already normalized/bounded, e.g. UCI HAR datasets
    already_normalized = dataset_class.already_normalized

    # Split into training/valid datasets
    train_data, train_labels, valid_data, valid_labels = \
        valid_split_modality(dataset.train_data, dataset.train_labels, seed=seed)

    # The multiple modality data is stored like (e.g. where xs = train_data):
    # xs = [(example 1 modality 1, example 2 modality 1, ...),
    #       (example 1 modality 2, example 2 modality 2, ...)]

    # Calculate normalization only on the training data
    if FLAGS.normalize != "none" and not already_normalized:
        normalization = calc_normalization_modality(train_data, FLAGS.normalize)

        # Apply the normalization to the training, validation, and testing data
        train_data = apply_normalization_modality(train_data, normalization)
        valid_data = apply_normalization_modality(valid_data, normalization)
        test_data = apply_normalization_modality(dataset.test_data, normalization)
    else:
        test_data = dataset.test_data

    # Saving
    write_modality(train_filename, train_data, train_labels)
    write_modality(valid_filename, valid_data, valid_labels)
    write_modality(test_filename, test_data, dataset.test_labels)


def main(argv):
    output_dir = os.path.join("datasets", "tfrecords")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all possible datasets we can generate, but only the single-modality
    # ones. We'll generate the multi-modality ones later.
    adaptation_problems = datasets.names(single_modality=True)

    # Save tfrecord files for each of the adaptation problems
    if FLAGS.parallel:
        # TensorFlow will error from all processes trying to use ~90% of the
        # GPU memory on all parallel jobs, which will fail, so do this on the
        # CPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        if FLAGS.jobs == 0:
            cores = None
        else:
            cores = FLAGS.jobs

        run_job_pool(save_dataset,
            [(d, output_dir) for d in adaptation_problems], cores=cores)
    else:
        for dataset_name in adaptation_problems:
            save_dataset(dataset_name, output_dir)


if __name__ == "__main__":
    app.run(main)

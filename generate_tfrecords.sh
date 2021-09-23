#!/bin/bash
#
# Generate tfrecord files for the datasets
#

# Make sure submodules are up to date
git submodule init
git submodule update

# Create synthetic data
/usr/bin/time python -m datasets.synthetic_datasets_normal

# Generate TFRecord files
/usr/bin/time python -m datasets.main --jobs=1 --debug

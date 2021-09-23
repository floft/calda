#!/bin/bash
#
# Download files from high performance cluster
#
. kamiak_config.sh

# Note both have trailing slashes
from="$remotessh:$remotedir"
to="$localdir"

rsync -Pahuv \
    --include="results*/" --include="results*/results_*.*" \
    --include="datasets/" --include="datasets/tfrecords/" --include="datasets/tfrecords/*.pickle*" \
    --exclude="*" "$from" "$to"

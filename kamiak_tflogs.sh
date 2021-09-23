#!/bin/bash
#
# Download the TF logs every once in a while to keep TensorBoard updated
# Then run: tensorboard  --logdir logs/
#
. kamiak_config.sh

# Note both have trailing slashes
from="$remotessh:$remotedir"
to="$localdir"

# Optionally pass in which suffix to sync. Useful if there's a lot of log
# folders you don't want to download.
[[ ! -e $1 ]] && suffix="-$1" || suffix=""

# Also, for further matching, allow custom folder matches. The folders are in
# the format <dataset>-<uid>-<method>-<debugnum>.
[[ ! -e $2 ]] && folder="$2" || folder=""

# TensorFlow logs
while true; do
    # --inplace so we don't get "file created after file even though it's
    #   lexicographically earlier" in TensorBoard, which basically makes it
    #   never update without restarting TensorBoard
    rsync -Pahuv --inplace \
        --include="$logFolder$suffix*/" --include="$logFolder$suffix*/$folder*" --include="$logFolder$suffix*/$folder*/*" \
        --exclude="*" --exclude="*/" "$from" "$to"
    sleep 30
done

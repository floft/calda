#!/bin/bash
#
# Upload files to high performance cluster
#
. kamiak_config.sh

# Note both have trailing slashes
from="$localdir"
to="$remotessh:$remotedir"

# Make SLURM log folder
ssh "$remotessh" "mkdir -p \"$remotedir/slurm_logs\""

# Copy only select files
rsync -Pahuv --exclude="__pycache__" \
    --include="./" --include="*.py" --include="*.sh" --include="*.srun" \
    --include="datasets/" --include="datasets/*" --include="datasets/tfrecords/*" \
    --include="datasets/multimodal/" --include="datasets/multimodal/*" \
    --include="datasets/as_images/" --include="datasets/as_images/*" --include="datasets/as_images/*/*" --include="datasets/as_images/*/*/*" \
    --include="*.tfrecord" --include="*.tar.gz" --include="*.zip" \
    --exclude="*" "$from" "$to"

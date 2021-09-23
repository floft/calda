#
# Config file for running on high performance cluster with Slurm
# Note: paths for rsync, so make sure all paths have a trailing slash
#
modelFolder="kamiak-models"
logFolder="kamiak-logs"
remotessh="kamiak"  # either what is in your .ssh/config file or user@hostname
project_name="$(basename "$(pwd)")"
remotedir="/path/to/remote/dir/${project_name}/"
localdir="/home/username/path/to/${project_name}/"

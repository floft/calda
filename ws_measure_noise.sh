#!/bin/bash
#
# Compute the true weak supervision noise values
# Usage: ./ws_measure_noise.sh | tee ws_measure_noise.txt
#
# Results:
#
# WISDM AR
# 0.05: 0.061817507433134
# 0.1: 0.118109909687866
# 0.2: 0.218323004498053
# 0.4: 0.375503798706614
#
# WISDM AT
# 0.05: 0.066680438290920
# 0.1: 0.127029761070465
# 0.2: 0.227170976015884
# 0.4: 0.401111629327313
#
compute() {
    train_jobid=$1
    cat slurm_logs/*${train_jobid}* | grep "Sum difference norm" | sed 's/.*: //g' | awk '{c++;sum+=$1}; END {avg=sum/c;printf "%.15f\n", avg}'
}

wisdm_ar_05=$(compute 29533574)
wisdm_ar_1=$(compute 29533594)
wisdm_ar_2=$(compute 29533603)
wisdm_ar_4=$(compute 29533609)

echo "WISDM AR"
echo "0.05: $wisdm_ar_05"
echo "0.1: $wisdm_ar_1"
echo "0.2: $wisdm_ar_2"
echo "0.4: $wisdm_ar_4"

wisdm_at_05=$(compute 29503586)
wisdm_at_1=$(compute 29503595)
wisdm_at_2=$(compute 29503596)
wisdm_at_4=$(compute 29503597)

echo "WISDM AT"
echo "0.05: $wisdm_at_05"
echo "0.1: $wisdm_at_1"
echo "0.2: $wisdm_at_2"
echo "0.4: $wisdm_at_4"

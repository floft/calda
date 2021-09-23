#!/bin/bash
#
# Verify everything still runs after code cleanup
#

printname() {
    echo -e "\n\n\n\n\n\n$1\n\n\n\n\n\n\n"
}

cleanup() {
    echo "Cleanup"
    rm kamiak_{train,eval}_experiments.srun
    rm kamiak_{train,eval}_tune.srun
    rm kamiak_{train,eval}_baseline_can_experiments.srun
    rm kamiak_{train,eval}_baseline_can_tune.srun
    rm results/results_test_example_best_target-ucihar-0-*.yaml
    rm results/results_test_example_best_target-wisdm_at-0-calda_any_r_ws.yaml
    rm results/results_test_example_best_target-*-0-calda_any_r.yaml
    rm -rf test-example-{logs,models}/
}

handle_terminate() {
    echo "Exiting"
    cleanup
    exit 1
}
trap "handle_terminate" SIGTERM SIGINT

checkreturn() {
    if (( $1 != 0 )); then
        echo "return value != 0"
        cleanup
        exit 1
    fi
}

# Dataset download/generation and conversion to tfrecord files
if [[ ! -e datasets/tfrecords/ucihar_1_train.tfrecord ]]; then
    if ! ./generate_tfrecords.sh; then
        echo "Could not generate tfrecord files"
        exit 1
    fi
fi

# Verify creating train/tune/eval scripts works
printname "output CALDA experiment scripts"
./experiments_msda.py --name experiments
checkreturn "$?"
printname "output CALDA tune scripts"
./experiments_msda.py --tune --name=tune
checkreturn "$?"

# Verify creating CAN train/tune/eval scripts works
printname "output CAN experiment scripts"
./experiments_msda.py --can --name experiments
checkreturn "$?"
printname "output CAN tune scripts"
./experiments_msda.py --can --tune --name=tune
checkreturn "$?"

# Verify each method can run
methods=(
    none upper codats
    calda_{xs,in,any}_{r,h,r_p,h_p}
    {codats,calda_{xs_h,any_r}}_ws
    {codats,sleep,aflac}_dg
    caldg_{xs_h,any_r}
    calda_{xs_h,any_r}_noadv
)
mkdir -p results
for i in "${methods[@]}"; do
    printname "train method $i"
    python3 main.py \
        --logdir=test-example-logs --modeldir=test-example-models \
        --method=$i --dataset=ucihar --sources=1,2 \
        --target=3 --uid=0 --debugnum=0 --gpumem=0 --steps=2
    checkreturn "$?"
    printname "eval method $i"
    python3 main_eval.py \
        --logdir=test-example-logs --modeldir=test-example-models \
        --jobs=1 --gpus=1 --gpumem=0 \
        --match="ucihar-0-$i-[0-9]*" --selection="best_target" \
        --output_file=results/results_test_example_best_target-ucihar-0-$i.yaml
    checkreturn "$?"
done

# Verify weak supervision noise works
printname "weak supervision noise - train"
python3 main.py \
    --logdir=test-example-logs --modeldir=test-example-models \
    --method=calda_any_r_ws --dataset=wisdm_at --sources=1,2 \
    --target=3 --uid=0 --debugnum=0 --gpumem=0 --ws_noise=0.4 --steps=2
checkreturn "$?"
printname "weak supervision noise - eval"
python3 main_eval.py \
    --logdir=test-example-logs --modeldir=test-example-models \
    --jobs=1 --gpus=1 --gpumem=0 \
    --match="wisdm_at-0-calda_any_r_ws-[0-9]*" --selection="best_target" \
    --output_file=results/results_test_example_best_target-wisdm_at-0-calda_any_r_ws.yaml
checkreturn "$?"

# Verify each dataset
datasets=(
    ucihar ucihhar wisdm_ar wisdm_at myo ninapro_db5_like_myo_noshift
    normal_n12_l3_inter0_intra1_5,0,0,0_sine
    normal_n12_l3_inter2_intra1_5,0,0,0_sine
    normal_n12_l3_inter0_intra1_0,0.5,0,0_sine
    normal_n12_l3_inter2_intra1_0,0.5,0,0_sine
    normal_n12_l3_inter1_intra0_0,0,5,0_sine
    normal_n12_l3_inter1_intra2_0,0,5,0_sine
    normal_n12_l3_inter1_intra0_0,0,0,0.5_sine
    normal_n12_l3_inter1_intra2_0,0,0,0.5_sine
)
for i in "${datasets[@]}"; do
    # pick valid target domains
    if [[ $i == myo ]]; then
        target=22
    elif [[ $i == normal_* ]]; then
        target=0
    else
        target=3
    fi

    printname "train dataset $i"
    python3 main.py \
        --logdir=test-example-logs --modeldir=test-example-models \
        --method=calda_any_r --dataset=$i --sources=1,2 \
        --target=$target --uid=0 --debugnum=0 --gpumem=0 --steps=2
    checkreturn "$?"
    printname "eval dataset $i"
    python3 main_eval.py \
        --logdir=test-example-logs --modeldir=test-example-models \
        --jobs=1 --gpus=1 --gpumem=0 \
        --match="$i-0-calda_any_r-[0-9]*" --selection="best_target" \
        --output_file=results/results_test_example_best_target-$i-0-calda_any_r.yaml
    checkreturn "$?"
done

printname "Success - all jobs were able to run"
cleanup

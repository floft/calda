# Contrastive Adversarial Learning for Multi-Source Time Series Domain Adaptation

Overview:

- Download data and convert to .tfrecord files for TensorFlow
  (*./generate_tfrecords.sh*)
- Train models (*main.py*)
- Evaluate models (*main_eval.py*)
- Analyze results (*analysis.py*)

## Installation

We used CUDA 10.1.105, CuDNN 7.6.4.38, Python 3.7.4, TensorFlow-GPU 2.2.0,
and PyTorch 1.9.0. We installed the following packages via *pip*:

    pip install --user --upgrade numpy cython
    pip install --user --upgrade tensorflow-gpu pillow lxml jupyter matplotlib \
        pandas scikit-learn scipy tensorboard tqdm pyyaml grpcio absl-py \
        tensorflow-addons torch torchvision easydict torchinfo pickle5

Note: the final few packages (e.g. PyTorch) are for the CAN baseline.

If you want to verify that everything is installed correctly, you can run the
`./test.sh` script that calls *generate_tfrecords.sh* followed by a number of
short experiments (~30 minutes total).

## Usage

### Example

Train a CALDA-XS,H model on person 1 and 2 of the UCI HAR dataset and adapt to person 3.

    python3 main.py \
        --logdir=example-logs --modeldir=example-models \
        --method=calda_xs_h --dataset=ucihar --sources=1,2 \
        --target=3 --uid=0 --debugnum=0 --gpumem=0

Monitor training progress:

    tensorboard --logdir example-logs

Then evaluate that model on the holdout test data, outputting the results to a
YAML file.

    mkdir -p results
    python3 main_eval.py \
        --logdir=example-logs --modeldir=example-models \
        --jobs=1 --gpus=1 --gpumem=0 \
        --match="ucihar-0-calda_xs_h-[0-9]*" --selection="best_target" \
        --output_file=results/results_example_best_target-ucihar-0-calda_xs_h.yaml

Specifically, look at *accuracy_task/target/validation* in the YAML result file
for the target domain test set accuracy.

### Full Experiments

If you want to run all of the experiments, you can generate the SLURM training
scripts:

    ./experiments_msda.py --name experiments

Then, run the *kamiak_{train,eval}_experiments.srun* on your SLURM cluster
after installing TensorFlow, etc. (some modification to these scripts may be
required to make this work on your cluster).

If you want to re-run hyperparameter tuning, you can generate the tuning
scripts as well, though the results of tuning are already included in
*hyperparameters.py*.

    ./experiments_msda.py --tune --name=tune

Note before using the SLURM scripts, you will need to update `kamiak_config.sh`
to the correct paths and `kamiak_tensorflow_gpu.sh` and
`kamiak_tensorflow_cpu.sh` to load the appropriate packages.

### Baselines

The No Adaptation and CoDATS baselines are included in this repository. However,
the CAN baseline is a fork of the code from their original paper. To run the
CAN baseline on the time series datasets:

- Download repositories
  - Clone this repo to `calda`
  - Clone the [CAN baseline repository](https://github.com/floft/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation)
  into `Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation`
  - Note: the SLURM scripts require these two directory names and them both having the same parent directory.
- Create pickle files for the datasets (*./generate_tfrecords_as_images.sh*)
- Run individual train/test as explained in the
  [CAN baseline repository](https://github.com/floft/Contrastive-Adaptation-Network-for-Unsupervised-Domain-Adaptation)
  instructions.

Alternatively, for full hyperparameter tuning (following same procedure as for CALDA described earlier), after optionally updating the hyperparameter set in `hyperparameter_tuning_experiments_can.py`. Note that the hyperparameters we found to be the best are already included in `hyperparameters.py`.

    ./experiments_msda.py --can --tune --name can_tune

For the full set of experiments, after hyperparameter tuning and analysis:

    ./experiments_msda.py --can --name can_tuned

This will generate the SLURM train/eval scripts for all of the CAN baseline experiments.

### Analysis

Then look at the resulting *results/results_\*.yaml* files or analyze with
*analysis.py*.

## Navigating the Code

In the paper we propose CALDA which has a variety of different variations that
can be chosen with the *--method=...* flag. The options are (e.g.,
corresponding to the names CALDA-XS,R, CALDA-In,R, etc. in the paper):

    - none
    - upper
    - codats
    - calda_xs_r
    - calda_in_r
    - calda_any_r
    - calda_xs_h
    - calda_in_h
    - calda_any_h
    - calda_xs_r_p
    - calda_in_r_p
    - calda_any_r_p
    - calda_xs_h_p
    - calda_in_h_p
    - calda_any_h_p
    - codats_ws
    - calda_xs_h_ws
    - calda_any_r_ws
    - codats_dg
    - sleep_dg
    - aflac_dg
    - caldg_xs_h
    - caldg_any_r
    - calda_xs_h_noadv
    - calda_any_r_noadv

The code for these methods is found in *methods.py:MethodCaldaBase*. The
function for computing the contrastive loss is *_similarity_loss()* within this
class.

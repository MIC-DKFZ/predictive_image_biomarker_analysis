#!/bin/bash
 
source ~/.bashrc
 
#export DATASET_LOCATION=/absolute/path/to/datasets
#export EXPERIMENT_LOCATION=/absolute/path/to/experiments
#export CONFIGS_LOCATION=/absolute/path/to/configs

conda activate predimgbmenv

echo "$@"

python3 ./predimgbmanalysis/train.py -m --config-path=$CONFIGS_LOCATION --config-name="config_nsclcradiomics_treatidx5fold.yaml" pl_params.data_params.b_prog=0.0,0.2,0.4,0.6,0.8,1.0 pl_params.data_params.b_pred=0.0,0.2,0.4,0.6,0.8,1.0 pl_params.k_fold_splits=5 pl_params.fold_idx=0,1,2,3,4 $@


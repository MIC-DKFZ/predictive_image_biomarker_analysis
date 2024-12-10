#!/bin/bash
 
source ~/.bashrc
 
#export DATASET_LOCATION=/absolute/path/to/datasets
#export EXPERIMENT_LOCATION=/absolute/path/to/experiments
#export CONFIGS_LOCATION=/absolute/path/to/configs

conda activate predimgbmenv

echo "$@"

#python3 ./predimgbmanalysis/train.py -m --config-path=$CONFIGS_LOCATION --config-name="config_cmnist.yaml" pl_params.model_type=miniresnet pl_params.mode=mtl_loss pl_params.data_params.prog_feature=b_digitcircle pl_params.data_params.pred_feature=b_col pl_params.data_params.b_prog=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 pl_params.data_params.b_pred=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 hydra.job.name=miniresnetmtl_cmnista +pl_params.data_params.pad_data=[[0,0],[2,2],[2,2]] $@

python3 ./predimgbmanalysis/train.py -m --config-path=$CONFIGS_LOCATION --config-name="config_cmnist.yaml" pl_params.data_params.b_prog=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 pl_params.data_params.b_pred=0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 +pl_params.data_params.pad_data=[[0,0],[2,2],[2,2]] $@



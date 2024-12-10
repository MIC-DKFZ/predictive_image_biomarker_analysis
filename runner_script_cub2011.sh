#!/bin/bash
 
source ~/.bashrc
 
#export DATASET_LOCATION=/absolute/path/to/datasets
#export EXPERIMENT_LOCATION=/absolute/path/to/experiments
#export CONFIGS_LOCATION=/absolute/path/to/configs

conda activate predimgbmenv

echo "$@"

#python3 ./predimgbmanalysis/train.py -m --config-path=$CONFIGS_LOCATION --config-name="config_cub2011.yaml" pl_params.model_type=resnet18 pl_params.mode=mtl_loss pl_params.data_params.prog_feature=b_colwhite pl_params.data_params.pred_feature=b_billlong pl_params.data_params.b_prog=0.0,0.2,0.4,0.6,0.8,1.0 pl_params.data_params.b_pred=0.0,0.2,0.4,0.6,0.8,1.0 hydra.job.name=resnet18mtl_cub2011a $@

python3 ./predimgbmanalysis/train.py -m --config-path=$CONFIGS_LOCATION --config-name="config_isic2018.yaml" pl_params.data_params.b_prog=0.0,0.2,0.4,0.6,0.8,1.0 pl_params.data_params.b_pred=0.0,0.2,0.4,0.6,0.8,1.0 $@

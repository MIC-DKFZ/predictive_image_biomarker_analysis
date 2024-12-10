import os

import numpy as np
import torch
import math
import plotly.graph_objects as go

from predimgbmanalysis.eval_biomarkers import *

from torch import nn

from pytorch_grad_cam.utils.model_targets import (
    ClassifierOutputTarget,
    RawScoresOutputTarget,
)
from notebooks.gradcam3D import GradCAM3D

import warnings

warnings.filterwarnings("ignore")


class TwoHead_Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        output = self.model(input)
        output = torch.hstack(list(output))
        return output


class CATE_Wrapper(nn.Module):
    # difference of two heads

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input):
        output = self.model(input)
        return output[1] - output[0]


network_drive_path = os.getenv("DATASET_LOCATION", "/absolute/path/to/datasets")
dataset_root = {
    "CMNIST": network_drive_path,
    "cub": os.path.join(network_drive_path, "CUB_200_2011"),
    "isic": os.path.join(network_drive_path, "ISIC2018/"),
    "lungCT": os.path.join(network_drive_path, "NSCLC_Radiomics/"),
}

experiments = os.getenv("EXPERIMENT_LOCATION", "/absolute/path/to/experiments")
experiment_dirs = {
    "CMNIST_a": os.path.join(
        experiments, "2022-10-25_toymodel_miniresnetmtl_cmnist3a_02"
    ),
    "CMNIST_b": os.path.join(
        experiments, "2022-10-25_toymodel_miniresnetmtl_cmnist3b_02"
    ),
    "CMNISTvline_a": os.path.join(
        experiments, "2022-10-31_toymodel_miniresnetmtl_cmnist5a"
    ),
    "CMNISTvline_b": os.path.join(
        experiments, "2022-10-31_toymodel_miniresnetmtl_cmnist5b"
    ),
    "cub_a": os.path.join(
        experiments, "2022-10-27_toymodel_resnet18mtl_cub2011allclassesa_01"
    ),
    "cub_b": os.path.join(
        experiments, "2022-10-27_toymodel_resnet18mtl_cub2011allclassesb_01"
    ),
    "isic_a": os.path.join(
        experiments, "2022-10-19_toymodel_resnet18mtl_isic2018a_binary"
    ),
    "isic_b": os.path.join(
        experiments, "2022-10-19_toymodel_resnet18mtl_isic2018b_binary"
    ),
    "isic_a_newarch": os.path.join(
        experiments, "2022-10-28_toymodel_resnet18mtl_isic2018a_binary_05"
    ),
    "isic_b_newarch": os.path.join(
        experiments, "2022-10-28_toymodel_resnet18mtl_isic2018b_binary_05"
    ),
    "lungCT_a_fold0": os.path.join(
        experiments,
        "test_nsclctumourpatchesnnunet_linear_zscore_mtl4fc_a_extendedrandtransformnew_fold0_complete",
    ),
    "lungCT_a_fold1": os.path.join(
        experiments,
        "test_nsclctumourpatchesnnunet_linear_zscore_mtl4fc_a_extendedrandtransformnew_fold1_complete",
    ),
    "lungCT_a_fold2": os.path.join(
        experiments,
        "test_nsclctumourpatchesnnunet_linear_zscore_mtl4fc_a_extendedrandtransformnew_fold2_complete",
    ),
    "lungCT_a_fold3": os.path.join(
        experiments,
        "test_nsclctumourpatchesnnunet_linear_zscore_mtl4fc_a_extendedrandtransformnew_fold3_complete",
    ),
    "lungCT_a_fold4": os.path.join(
        experiments,
        "test_nsclctumourpatchesnnunet_linear_zscore_mtl4fc_a_extendedrandtransformnew_fold4_complete",
    ),
    "lungCT_b_fold0": os.path.join(
        experiments,
        "test_nsclctumourpatchesnnunet_linear_zscore_mtl4fc_b_extendedrandtransformnew_fold0_complete",
    ),
    "lungCT_b_fold1": os.path.join(
        experiments,
        "test_nsclctumourpatchesnnunet_linear_zscore_mtl4fc_b_extendedrandtransformnew_fold1_complete",
    ),
    "lungCT_b_fold2": os.path.join(
        experiments,
        "test_nsclctumourpatchesnnunet_linear_zscore_mtl4fc_b_extendedrandtransformnew_fold2_complete",
    ),
    "lungCT_b_fold3": os.path.join(
        experiments,
        "test_nsclctumourpatchesnnunet_linear_zscore_mtl4fc_b_extendedrandtransformnew_fold3_complete",
    ),
    "lungCT_b_fold4": os.path.join(
        experiments,
        "test_nsclctumourpatchesnnunet_linear_zscore_mtl4fc_b_extendedrandtransformnew_fold4_complete",
    ),
    "lungCT_a": os.path.join(
        experiments,
        "test_nsclctumourpatchesnnunet_linear_zscore_mtl4fc_a_extendedrandtransformnewfulltraincv_complete",
    ),
    "lungCT_b": os.path.join(
        experiments,
        "test_nsclctumourpatchesnnunet_linear_zscore_mtl4fc_b_extendedrandtransformnewfulltraincv_complete",
    ),
}


log_names = {
    "CMNIST_a": [
        "2022-10-27_13-05-44_120_bpred=1.0_bprog=1.0_finalact=None",
        "2022-10-27_04-03-14_96_bpred=0.8_bprog=0.8_finalact=None",
    ],
    "CMNIST_b": [
        "2022-10-26_21-51-30_120_bpred=1.0_bprog=1.0_finalact=None",
        "2022-10-26_15-56-12_96_bpred=0.8_bprog=0.8_finalact=None",
    ],
    "CMNISTvline_a": [
        "2022-11-02_11-35-26_120_bpred=1.0_bprog=1.0_finalact=None",
        "2022-11-01_21-32-49_84_bpred=0.7_bprog=0.7_finalact=None",
        "2022-11-01_12-13-19_60_bpred=0.5_bprog=0.5_finalact=None",
    ],
    "CMNISTvline_b": [
        "2022-11-02_11-47-04_120_bpred=1.0_bprog=1.0_finalact=None",
        "2022-11-01_21-42-48_84_bpred=0.7_bprog=0.7_finalact=None",
        "2022-11-01_12-20-51_60_bpred=0.5_bprog=0.5_finalact=None",
    ],
    "cub_a": [
        "2022-10-30_07-41-48_35_bpred=1.0_bprog=1.0_finalact=None",
        "2022-10-29_21-04-59_28_bpred=0.8_bprog=0.8_finalact=None",
        "2022-10-29_09-31-50_21_bpred=0.6_bprog=0.6_finalact=None",
    ],
    "cub_b": [
        "2022-10-30_08-12-09_35_bpred=1.0_bprog=1.0_finalact=None",
        "2022-10-29_21-35-28_28_bpred=0.8_bprog=0.8_finalact=None",
        "2022-10-29_09-59-09_21_bpred=0.6_bprog=0.6_finalact=None",
    ],
    "isic_a": [
        "2022-10-23_17-37-07_35_bpred=1.0_bprog=1.0_finalact=None",
        "2022-10-23_03-42-00_28_bpred=0.8_bprog=0.8_finalact=None",
        "2022-10-22_12-42-59_21_bpred=0.6_bprog=0.6_finalact=None",
    ],
    "isic_b": [
        "2022-10-23_19-03-41_35_bpred=1.0_bprog=1.0_finalact=None",
        "2022-10-23_05-53-30_28_bpred=0.8_bprog=0.8_finalact=None",
        "2022-10-22_15-42-35_21_bpred=0.6_bprog=0.6_finalact=None",
    ],
    "isic_a_newarch": [
        "2022-11-01_04-19-04_35_bpred=1.0_bprog=1.0_finalact=None",
        "2022-10-31_12-00-21_28_bpred=0.8_bprog=0.8_finalact=None",
        "2022-10-30_19-40-27_21_bpred=0.6_bprog=0.6_finalact=None",
    ],
    "isic_b_newarch": [
        "2022-10-30_07-50-50_35_bpred=1.0_bprog=1.0_finalact=None",
        "2022-10-30_01-22-52_28_bpred=0.8_bprog=0.8_finalact=None",
        "2022-10-29_17-37-53_21_bpred=0.6_bprog=0.6_finalact=None",
    ],
    "lungCT_a_fold0": [
        "2024-01-26_14-28-16_35_bpred=1.0_bprog=1.0_finalact=None_kfold_idx=0",
        "2024-01-25_03-39-05_28_bpred=0.8_bprog=0.8_finalact=None_kfold_idx=0",
    ],
    "lungCT_a_fold1": [
        "2024-01-29_19-17-43_0_bpred=1.0_bprog=1.0_finalact=None_kfold_idx=1",
        "2024-01-30_19-53-21_4_bpred=0.8_bprog=0.8_finalact=None_kfold_idx=1",
    ],
    "lungCT_a_fold2": [
        "2024-01-25_19-45-39_35_bpred=1.0_bprog=1.0_finalact=None_kfold_idx=2",
        "2024-01-24_13-26-30_28_bpred=0.8_bprog=0.8_finalact=None_kfold_idx=2",
    ],
    "lungCT_a_fold3": [
        "2024-01-26_16-34-54_35_bpred=1.0_bprog=1.0_finalact=None_kfold_idx=3",
        "2024-01-25_04-24-35_28_bpred=0.8_bprog=0.8_finalact=None_kfold_idx=3",
    ],
    "lungCT_a_fold4": [
        "2024-01-28_06-23-02_5_bpred=1.0_bprog=1.0_finalact=None_kfold_idx=4",
        "2024-01-28_20-20-36_0_bpred=0.8_bprog=0.8_finalact=None_kfold_idx=4",
    ],
    "lungCT_b_fold0": [
        "2024-01-25_19-48-28_35_bpred=1.0_bprog=1.0_finalact=None_kfold_idx=0",
        "2024-01-24_13-29-39_28_bpred=0.8_bprog=0.8_finalact=None_kfold_idx=0",
    ],
    "lungCT_b_fold1": [
        "2024-01-31_04-00-52_17_bpred=1.0_bprog=1.0_finalact=None_kfold_idx=1",
        "2024-01-30_00-06-48_13_bpred=0.8_bprog=0.8_finalact=None_kfold_idx=1",
    ],
    "lungCT_b_fold2": [
        "2024-01-29_14-41-45_0_bpred=1.0_bprog=1.0_finalact=None_kfold_idx=2",
        "2024-01-30_14-13-38_4_bpred=0.8_bprog=0.8_finalact=None_kfold_idx=2",
    ],
    "lungCT_b_fold3": [
        "2024-01-31_14-17-02_11_bpred=1.0_bprog=1.0_finalact=None_kfold_idx=3",
        "2024-01-30_20-09-55_8_bpred=0.8_bprog=0.8_finalact=None_kfold_idx=3",
    ],
    "lungCT_b_fold4": [
        "2024-02-02_05-20-30_17_bpred=1.0_bprog=1.0_finalact=None_kfold_idx=4",
        "2024-02-01_04-58-41_13_bpred=0.8_bprog=0.8_finalact=None_kfold_idx=4",
    ],
    "lungCT_a": [
        "2024-01-26_22-15-32_0_bpred=1.0_bprog=1.0_finalact=None_kfold_idx=None",
        "2024-01-28_16-10-15_4_bpred=0.8_bprog=0.8_finalact=None_kfold_idx=None",
    ],
    "lungCT_b": [
        "2024-01-26_21-54-06_0_bpred=1.0_bprog=1.0_finalact=None_kfold_idx=None",
        "2024-01-29_01-23-41_4_bpred=0.8_bprog=0.8_finalact=None_kfold_idx=None",
    ],
}


name = "lungCT_a"
data_name = "lungCT"
(
    model_lungCT_a,
    dl_lungCT_a,
    bprog_lungCT_a,
    bpred_lungCT_a,
    dl_lungCT_seg_a,
) = get_interpretation(
    experiment_dir=experiment_dirs[name],
    use_cuda=False,
    n_batch=300,
    log_name=log_names[name][1],
    get_saliency_maps=False,
    dataset_root=dataset_root[data_name],
    env="test",
    get_segmentation_dl=True,
)

name = "lungCT_b"
data_name = "lungCT"
model_lungCT_b, dl_lungCT_b, bprog_lungCT_b, bpred_lungCT_b = get_interpretation(
    experiment_dir=experiment_dirs[name],
    use_cuda=False,
    n_batch=83,
    log_name=log_names[name][1],
    get_saliency_maps=False,
    dataset_root=dataset_root[data_name],
    env="test",
)


seg_lungCT = next(iter(dl_lungCT_seg_a)).numpy()
data_lungCT = next(iter(dl_lungCT_a))
data_lungCT = torch.Tensor(data_lungCT[0]).to("cuda:0")


n = 8  # Add Idx number here

# GradCAM

attr_gcam_lungCT = []
for model in (model_lungCT_a, model_lungCT_b):
    attr_method = GradCAM3D(
        model=TwoHead_Wrapper(model),
        target_layers=[
            TwoHead_Wrapper(model).model.model.layer1,
            TwoHead_Wrapper(model).model.model.layer2,
            TwoHead_Wrapper(model).model.model.layer3,
        ],
    )
    for target in (0, 1):
        attr_values = attr_method(
            input_tensor=data_lungCT[n].unsqueeze(0),
            targets=[ClassifierOutputTarget(target)],
        )
        attr_gcam_lungCT.append(attr_values.squeeze(0))

for model in (model_lungCT_a, model_lungCT_b):
    attr_method = GradCAM3D(
        model=CATE_Wrapper(model),
        target_layers=[
            CATE_Wrapper(model).model.model.layer1,
            CATE_Wrapper(model).model.model.layer2,
            CATE_Wrapper(model).model.model.layer3,
        ],
    )
    attr_values = attr_method(
        input_tensor=data_lungCT[n].unsqueeze(0), targets=[RawScoresOutputTarget()]
    )
    attr_gcam_lungCT.append(attr_values.squeeze(0))

attr_gcam_lungCT = [attr_gcam_lungCT[i] for i in [0, 1, 4, 2, 3, 5]]

model_type = [
    "TwoHead_0_a",
    "TwoHead_1_a",
    "CATE_a",
    "TwoHead_0_b",
    "TwoHead_1_b",
    "CATE_b",
]

if not os.path.exists("./Images/lungCT/" + str(n) + "/3D/"):
    os.makedirs("./Images/lungCT/" + str(n) + "/3D/")

cmap = [[0, "white"], [0.5, "red"], [1, "red"]]


def NormalizeData(data):
    return (data - np.min(data)) / ((np.max(data) - np.min(data)) + 0.00000000001)


X, Y, Z = np.mgrid[0:162:162j, 0:162:162j, 0:54:54j]

for i in range(6):
    print("GCAM: " + str(i + 1) + "/6")
    fig = go.Figure()

    fig.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=NormalizeData(np.moveaxis(seg_lungCT[n, 0], 0, 2).flatten()),
            isomin=0.1,
            isomax=0.8,
            opacity=0.1,
            surface_count=10,  # needs to be a large number for good volume rendering
            surface_show=True,
            showscale=False,
            colorscale="grey",
        )
    )

    fig.update_scenes(
        xaxis_title="",
        yaxis_title="",
        zaxis_title="",
        aspectmode="cube",
    )

    camera = dict(eye=dict(x=1.8, y=1.8, z=0.9))

    fig.update_layout(
        autosize=False,
        template="plotly_white",
        scene_camera=camera,
        height=400,
        width=400,
        margin=dict(
            t=0, r=0, l=0, b=0
        ),  # left margin  # right margin  # bottom margin  # top margin
        font=dict(family="Helvetica", color="#000000", size=11),
    )

    fig.add_trace(
        go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=NormalizeData(np.abs(attr_gcam_lungCT[i]).flatten()),
            isomin=0.4,
            isomax=1.0,
            # opacity=0.15,
            surface_count=10,  # needs to be a large number for good volume rendering
            showscale=False,
            colorscale="jet",
            opacityscale=[
                [0, 0.2],
                [0.2, 0.4],
                [0.4, 0.5],
                [0.6, 0.7],
                [0.8, 0.8],
                [1, 1],
            ],
            surface_show=True,
        ),
    )

    fig.write_image(
        "./Images/lungCT/" + str(n) + "/3D/GCAM_" + model_type[i] + ".png", scale=2
    )

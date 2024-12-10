import os
import torch
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import math

# from argparse import ArgumentParser
import pytorch_lightning as pl
from pytorch_lightning.core.saving import save_hparams_to_yaml
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torchvision.datasets.utils as dataset_utils
from torchvision.datasets.folder import default_loader
from torchvision.transforms.functional import to_tensor
from sklearn.model_selection import train_test_split

from predimgbmanalysis.augmentation import (
    get_batchgenerators_transforms,
    get_monai_transforms,
)

logger = logging.getLogger(__name__)


def get_treatment_var_sample(env: str, n: int, seed=0):
    torch.manual_seed(seed)

    if (env is None) or (env in ["train", "val", "test", "traincv", "traincvval"]):
        treat = torch.randint(low=0, high=2, size=(n,))
    else:
        if isinstance(env, str):
            if env.startswith("EG"):
                treat = torch.ones((n))
            elif env.startswith("CG"):
                treat = torch.zeros((n))
            else:
                logger.warning("unknown env, use random env instead")
                treat = torch.randint(low=0, high=2, size=(n,))
        elif isinstance(env, int):
            treat = torch.ones((n)) * env
        else:
            logger.warning("unknown env, use random env instead")
            treat = torch.randint(low=0, high=2, size=(n,))

    return treat


def get_biomarker_sample(
    input_type: str,
    n: int,
    input_dim: int,
    input_dim_split=1,
    x_input=None,
    custom_feature_idx=None,
    seed=0,
):
    assert input_type in [
        "custom",
        "binomial",
        "uniform",
        "normal",
        "uniform_centralised",
        "binomial_centralised",
        "binomial_uniform",
        "uniform_binomial",
    ]
    torch.manual_seed(seed)
    if input_type == "custom" and (x_input is not None):
        assert x_input.shape == (n, input_dim)
        x = x_input
    elif input_type == "binomial":
        x = torch.randint(low=0, high=2, size=(n, input_dim))
    elif input_type == "uniform":
        x = torch.rand(n, input_dim)
    elif input_type == "normal":
        x = torch.randn(n, input_dim)
    elif input_type == "uniform_centralised":
        x = torch.rand(n, input_dim) * 2 - 1.0
    elif input_type == "binomial_centralised":
        x = torch.randint(low=0, high=2, size=(n, input_dim)) * 2 - 1
    elif input_type == "binomial_uniform":
        x = torch.zeros((n, input_dim))
        x[:, :input_dim_split] = torch.randint(low=0, high=2, size=(n, input_dim_split))
        x[:, input_dim_split:] = torch.rand(n, input_dim - input_dim_split)
    elif input_type == "uniform_binomial":
        x = torch.zeros((n, input_dim))
        x[:, :input_dim_split] = torch.rand(n, input_dim_split)
        x[:, input_dim_split:] = torch.randint(
            low=0, high=2, size=(n, input_dim - input_dim_split)
        )

    if input_type != "custom" and (custom_feature_idx is not None):
        assert x_input.shape[0] == n
        # assert custom_feature_idx < input_dim
        x = x.float()
        x[:, custom_feature_idx] = x_input.float()
    elif (x_input is not None) and (custom_feature_idx is None):
        print("x_input given, but no custom_feature_idx")
    return x


def get_data_model_output(data_model_type: str, b: list, c0: float, x, treat):
    if data_model_type == "simple":
        # f = b0 + b1*T + b2*x_prog + b3*x_pred*T
        f = b[0] + b[1] * treat + b[2] * x[:, 0] + b[3] * x[:, 1] * treat
    elif data_model_type == "threshold":
        # f = b0 + b1*T + b2*x_prog + b3*x_pred*T
        f = (
            b[0]
            + b[1] * treat
            + b[2] * x[:, 0]
            + b[3] * torch.heaviside(x[:, 1] - c0, values=torch.tensor([0.0])) * treat
        )
    elif data_model_type == "full":
        # f = b0 + b1*T + b2*x0 + b3*x1 + (b4*x0 + b5*x1)*T
        f = (
            b[0]
            + b[1] * treat
            + b[2] * x[:, 0]
            + b[3] * x[:, 1]
            + (b[4] * x[:, 0] + b[5] * x[:, 1]) * treat
        )
    else:
        logger.warning(
            "Unknown data_model_type={data_model_type}. Choosing default: full model"
        )
        # f = b0 + b1*T + b2*x0 + b3*x1 + (b4*x0 + b5*x1)*T
        f = (
            b[0]
            + b[1] * treat
            + b[2] * x[:, 0]
            + b[3] * x[:, 1]
            + (b[4] * x[:, 0] + b[5] * x[:, 1]) * treat
        )
    return f


def example_linear(
    env=None,
    n=10000,
    c0=None,
    b=[0.0, 0.0, 0.5, 0.1, 0.1, 0.5],
    input_type="binomial",  # {"binomial", "uniform", "normal", "binomial_uniform", "uniform_binomial", "custom"}
    data_model_type="full",  # {"simple", "threshold", full"}
    input_dim=2,
    input_dim_split=1,
    x_input=None,
    custom_feature_idx=None,
    get_counterfactuals=False,
    seed=0,
    get_treatment=None,
    **kwargs,
):
    if data_model_type == "full":
        assert len(b) == 6, "len(b) must be 6"
    if data_model_type == "threshold":
        assert c0 is not None, "specify threshold c0"
    assert input_dim >= 2

    # treatment variable
    treat = get_treatment_var_sample(env, n, seed=seed)

    # prognostic and predictive biomarker (additional dimensions: noise)
    x = get_biomarker_sample(
        input_type,
        n,
        input_dim,
        input_dim_split,
        x_input,
        custom_feature_idx,
        seed=seed,
    )

    f = get_data_model_output(data_model_type, b, c0, x, treat)

    # generate outputs by logit[P(Y = 1|T,X)] = f(X, T);
    target = f
    output = (x.float(), target[:, None])

    if get_treatment is None:
        if (
            isinstance(env, int)
            and env not in {0, 1}
            or isinstance(env, str)
            and not env.startswith(("EG", "CG"))
        ):
            get_treatment = True
        else:
            get_treatment = False

    if get_treatment:
        output += (treat,)
    if get_counterfactuals:
        f_counterf = get_data_model_output(data_model_type, b, c0, x, 1.0 - treat)
        output = output + (f_counterf[:, None],)
    return output


get_data_fun = {
    "linear": example_linear,
}


def load_img_data_cmnist(
    env,
    n_batch,
    root,
    data_fun_name="linear",
    pred_feature=None,
    prog_feature=None,
    normalise_range=False,
    normalise_value=None,
    center_range=False,
    # img_size=28,
    bg_value=None,
    train=True,
    save_num_data_dir=None,
    val_split_size=0.2,
    get_1d_data=False,
    get_counterfactuals=False,
    pad_data=None,
    env_treatidx_mapping={"CG": 0, "EG": 1},
    **data_kwargs,
):
    feature_type_list = [
        "b_col",
        "b_bgcol",
        "b_digit",
        "b_digitcircle",
        "c_colred",
        "c_colgreen",
    ]
    if prog_feature is None:
        prog_feature = "c_colred"
    if pred_feature is None:
        pred_feature = "c_colgreen"
    logger.debug(
        f"Preparing colored MNIST data with prognostic feature {prog_feature} \
                and predictive feature {pred_feature}, train/val split={1-val_split_size}/{val_split_size}."
    )

    # assert env in ["CG", "EG", "train", "CGval", "EGval", "val", "test"]
    assert pred_feature in feature_type_list
    assert prog_feature in feature_type_list
    assert pred_feature != prog_feature
    assert (val_split_size >= 0) and (
        val_split_size <= 1
    ), "val_split size must be in range [0,1]"
    if ("digit" in pred_feature) or ("digit" in prog_feature):
        assert (pred_feature.find("digit") > -1) != (
            prog_feature.find("digit") > -1
        ), f"Conflict of features {(pred_feature, prog_feature)}, cannot both be digit features"
    if pred_feature.startswith("b") or prog_feature.startswith("b"):
        assert "binomial" in data_kwargs.get(
            "input_type"
        ), f"Binary data must be sampled from binomial distribution, instead sampled from {data_kwargs.get('input_type')}"
    elif pred_feature.startswith("c") or prog_feature.startswith("c"):
        assert ("centralised" in data_kwargs.get("input_type")) or (
            "uniform" in data_kwargs.get("input_type")
        ), f"Continuous data must be sampled from continuous distribution, instead sampled from {data_kwargs.get('input_type')}"

    mnist = datasets.mnist.MNIST(root, train=train, download=False)
    mnist_array = mnist.data.numpy()
    mnist_targets_array = mnist.targets.numpy()

    if pad_data is not None:
        mnist_array = np.pad(mnist_array, pad_width=pad_data)

    if train:
        if not "traincv" in env:
            X_train, X_val, y_train, y_val = train_test_split(
                mnist_array,
                mnist_targets_array,
                test_size=val_split_size,
                random_state=1,
            )
            if env in ["CG", "EG", "train"]:
                mnist_array = X_train
                mnist_targets_array = y_train
            elif env in ["CGval", "EGval", "val"]:
                mnist_array = X_val
                mnist_targets_array = y_val

        # environment (=treatment group) split: split dataset into two even sized parts
        n_split = int(len(mnist_targets_array) / 2)
        if env.startswith("CG"):
            mnist_array = mnist_array[:n_split]
            mnist_targets_array = mnist_targets_array[:n_split]
        elif env.startswith("EG"):
            mnist_array = mnist_array[n_split:]
            mnist_targets_array = mnist_targets_array[n_split:]

    red_colour = torch.tensor([1, 0, 0], dtype=torch.uint8)
    green_colour = torch.tensor([0, 1, 0], dtype=torch.uint8)

    if n_batch is not None:
        assert n_batch <= len(mnist)

    if pred_feature.startswith("b_digit") or prog_feature.startswith("b_digit"):
        idx = 0 if pred_feature.startswith("b_digit") else 1
        feature = pred_feature if pred_feature.startswith("b_digit") else prog_feature
        if feature == "b_digit":  # only get digits 0 and 1
            if n_batch is not None:
                assert (
                    len(mnist_targets_array[mnist_targets_array < 2]) >= n_batch
                ), f"Error, n_batch={n_batch}<{len(mnist_targets_array[mnist_targets_array < 2])}"
            x = mnist_targets_array[mnist_targets_array < 2]
            mnist_array = mnist_array[mnist_targets_array < 2]
            mnist_targets_array = x
        if feature == "b_digitcircle":  # "has no circle"
            circledigits = [0, 6, 8, 9]
            x = mnist_targets_array
            x[np.isin(mnist_targets_array, circledigits)] = 0
            x[~np.isin(mnist_targets_array, circledigits)] = 1
        if feature == "b_digitvline":  # "has vertical line"
            vlinedigits = [1, 4, 5, 7, 9]
            x = mnist_targets_array
            x[~np.isin(mnist_targets_array, vlinedigits)] = 0
            x[np.isin(mnist_targets_array, vlinedigits)] = 1

        if n_batch is None:
            n_batch = len(mnist_targets_array)
        selected_env = next(
            (v for k, v in env_treatidx_mapping.items() if env.startswith(k)), env
        )
        data = get_data_fun[data_fun_name](
            env=selected_env,
            n=n_batch,
            input_dim=3,
            x_input=torch.from_numpy(x[:n_batch]),
            custom_feature_idx=idx,
            get_counterfactuals=get_counterfactuals,
            **data_kwargs,
        )
    else:
        if n_batch is None:
            n_batch = len(mnist_targets_array)
        selected_env = next(
            (v for k, v in env_treatidx_mapping.items() if env.startswith(k)), env
        )
        data = get_data_fun[data_fun_name](
            env=selected_env,
            n=n_batch,
            input_dim=3,
            get_counterfactuals=get_counterfactuals,
            **data_kwargs,
        )

    if len(data) == 2:
        x, y = data
    elif len(data) == 3:
        x, y, treat = data
    elif len(data) == 4:
        x, y, treat, y_counterf = data

    if save_num_data_dir is not None:  # saving numerical/tabular data only
        torch.save(data, os.path.join(save_num_data_dir, "numdata_train.pt"))
    mnist_array = np.repeat(mnist_array[:, np.newaxis], 3, axis=1)

    img = torch.from_numpy(mnist_array[:n_batch].copy())

    if "b_col" in (pred_feature, prog_feature):
        idx = 0 if pred_feature == "b_col" else 1
        img[x[:, idx] <= 0] = img[x[:, idx] <= 0] * red_colour[None, :, None, None]
        img[x[:, idx] > 0] = img[x[:, idx] > 0] * green_colour[None, :, None, None]

    if "b_bgcol" in (pred_feature, prog_feature):
        idx = 0 if pred_feature == "b_bgcol" else 1
        img_copy = img.clone()
        img[:, 0][x[:, idx] <= 0] = 255  # red_idx
        img[:, 1][x[:, idx] > 0] = 255  # green_idx
        img[(img_copy.sum(1) > 0).unsqueeze(1).expand(-1, 3, -1, -1)] = img_copy[
            (img_copy.sum(1) > 0).unsqueeze(1).expand(-1, 3, -1, -1)
        ]

    # Normalise image to range [0,1] before introducing continuous features
    img = img.float()
    img = (img - img.min()) / (img.max() - img.min())

    active_channels = [0, 0, 0]
    if "c_colred" in (pred_feature, prog_feature):
        assert "c_colredbg" not in (
            pred_feature,
            prog_feature,
        ), f"Conflict of foreground and background features {(pred_feature, prog_feature)}"
        idx = 0 if pred_feature == "c_colred" else 1

        img[:, 0] = img[:, 0] * x[:, idx][:, None, None]  # red_idx
        active_channels[0] = 1

    if "c_colgreen" in (pred_feature, prog_feature):
        assert "c_colgreenbg" not in (
            pred_feature,
            prog_feature,
        ), f"Conflict of foreground and background features {(pred_feature, prog_feature)}"

        idx = 0 if pred_feature == "c_colgreen" else 1
        img[:, 1] = img[:, 1] * x[:, idx][:, None, None]  # green_idx
        active_channels[1] = 1

    if sum(active_channels) > 0:
        img = (
            img * torch.tensor(active_channels)[None, :, None, None]
        )  # set overall non-active channels to 0

    if normalise_range:
        if bg_value is None:
            bg_value = (img.max() - img.min()) / 2
        img = 2 * (img - bg_value)  # [0,1] to [-1,1]

    if normalise_value is not None:
        assert (
            len(normalise_value) == 2
        ), f"normalise_value must be a list of [mean, std]={normalise_value}, got  input of type {type({normalise_value})}"
        logger.debug(
            f"Normalise data by mean {normalise_value[0]} and std {normalise_value[1]}."
        )
        img = (img - normalise_value[0]) / normalise_value[1]

    if center_range:
        img = img - img.mean()

    output = (img, y)
    if len(data) >= 3:
        treat = treat[: len(x)]
        output = output + (treat,)
    if get_1d_data:
        output = output + (x,)
    if get_counterfactuals:
        output = output + (y_counterf,)
    return output


# %% Dataset and Dataloaders


class ColoredMNIST(datasets.VisionDataset):
    """
    source: http://yann.lecun.com/exdb/mnist/

    Args:
        root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
        env (string): Which environment to load. Must be 1 of 'CG' (control group), 'EG' (experimental group), 'test', or 'all_train'.
        train (bool): Specifying training mode
        transform (callable, optional): A function/transform that returns a transformed version of an image
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        pred_feature (str): Dataset attribute set as the predictive biomarker feature, e.g. "b_col", "b_digitcircle"
        prog_feature (str): Dataset attribute set as the prognostic biomarker feature, e.g. "b_col", "b_digitcircle"
        class_ids (list, optional): If specified: which classes from the dataset to include
        pad_transform (int, optional): If specified: size of zero-padding
        load_1d_covariates (bool, optional): Whether to also return the covariates used to simulate the target outcomes
        get_counterfactuals (bool, optional): Whether to also return the ground truth counterfactual outcomes with the target
        treatidx_env_mapping (dict, optional): A dictionary mapping treatment indices to environment labels, e.g., {0: "CG", 1: "EG"}

    """

    def __init__(
        self,
        root="/absolute/path/to/datasets",
        env="CG",
        train=True,
        transform=None,
        target_transform=None,
        pred_feature=None,
        prog_feature=None,
        pad_transform=None,
        load_1d_covariates=False,
        get_counterfactuals=False,
        treatidx_env_mapping={0: "CG", 1: "EG"},
        **data_kwargs,
    ):
        if (
            (env in ["test", "val", "CGval", "EGval"])
            or (env.endswith("val"))
            or (not train)
        ):
            if transform == "randomspatial_transform":
                logger.info(
                    "Set default transform for test/validation data or if not in training mode."
                )
                transform = None

        if (transform is not None) and isinstance(transform, str):
            if transform == "randomspatial_transform":
                transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomApply(
                            [transforms.RandomRotation((90, 90))], p=0.5
                        ),
                    ]
                )
            elif transform == "resize":
                transform = transforms.Compose([transforms.Resize((224, 224))])

        super(ColoredMNIST, self).__init__(
            root, transform=transform, target_transform=target_transform
        )
        if pad_transform is not None and transform is None:
            self.transform = transforms.Pad(
                padding=[
                    pad_transform,
                ],
                padding_mode="edge",
            )
        self.env = env
        self.train = train
        self.pred_feature = pred_feature
        self.prog_feature = prog_feature
        self.load_1d_covariates = load_1d_covariates
        self.get_counterfactuals = get_counterfactuals
        self.x = None
        self.y_counterf = None
        self.treatidx_env_mapping = treatidx_env_mapping
        # swap values and keys for mapping
        self.env_treatidx_mapping = {v: k for k, v in treatidx_env_mapping.items()}
        logger.info(f"Loading ColoredMNIST dataset for environment {env}.")
        self.data, self.target = self.load_data(
            pred_feature=self.pred_feature,
            prog_feature=self.prog_feature,
            **data_kwargs,
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]
        target = self.target[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def load_data(self, pred_feature=None, prog_feature=None, **data_kwargs):
        data, *target = load_img_data_cmnist(
            env=self.env,
            n_batch=None,
            root=self.root,
            train=(self.env != "test"),
            pred_feature=pred_feature,
            prog_feature=prog_feature,
            get_1d_data=self.load_1d_covariates,
            get_counterfactuals=self.get_counterfactuals,
            env_treatidx_mapping=self.env_treatidx_mapping,
            **data_kwargs,
        )
        if self.get_counterfactuals:
            (*target, y_counterf) = target
            self.y_counterf = y_counterf.squeeze(1)
        if self.load_1d_covariates:
            (*target, x) = target
            self.x = x
        if len(target) == 1:
            target = target[0].squeeze(1)
        elif len(target) == 2:
            target = torch.stack((target[0].squeeze(1), target[1]), dim=1)
        else:
            logger.warning(f"More than two targets received: {target}.")
        return data, target


class CUB2011(datasets.VisionDataset):
    """
    source: http://www.vision.caltech.edu/datasets/cub_200_2011/

    Args:
        root (string): Root directory of dataset where ``CUB_200_2011/*.pt`` will exist.
        env (string): Which environment to load. Must be 1 of 'CG' (control group), 'EG' (experimental group), 'test', or 'all_train'.
        train (bool): Specifying training mode
        transform (Union[callable, str], optional): A string out of ["randomspatial_transform", "spatial_transform"] or function/transform that returns a transformed version of an image
        target_transform (callable, optional): A function/transform that takes returns a transformed version of a target
        pred_feature (str): Dataset attribute set as the predictive biomarker feature, e.g. "b_colwhite", "b_billlong"
        prog_feature (str): Dataset attribute set as the prognostic biomarker feature, e.g. "b_colwhite", "b_billlong"
        class_ids (list, optional): If specified: which classes from the dataset to include
        preload_imgs (str, optional): "original" for loading normalised tensor images, /not None/ for loading cropped images
        load_1d_covariates (bool, optional): Whether to also return the covariates used to simulate the target outcomes
        get_counterfactuals (bool, optional): Whether to also return the ground truth counterfactual outcomes with the target
        treatidx_env_mapping (dict, optional): A dictionary mapping treatment indices to environment labels, e.g., {0: "CG", 1: "EG"}
    """

    def __init__(
        self,
        root="/absolute/path/to/datasets/CUB_200_2011/CUB_200_2011/",
        env="CG",
        train=True,
        transform=None,
        target_transform=None,
        pred_feature=None,
        prog_feature=None,
        class_ids=None,
        preload_imgs=None,
        load_1d_covariates=False,
        get_counterfactuals=False,
        treatidx_env_mapping={0: "CG", 1: "EG"},
        **data_kwargs,
    ):
        if (
            (env in ["test", "val", "CGval", "EGval"])
            or (env.endswith("val"))
            or (not train)
        ):
            logger.info(
                "Set default transform for test/validation data or if not in training mode."
            )
            transform = None if preload_imgs is not None else "spatial_transform"
            if preload_imgs is not None:
                preload_imgs = True  # use preprocessed/transformed data instead.
                logger.info("Using preprocessed data.")
        if (transform is not None) and isinstance(transform, str):
            if transform == "randomspatial_transform":
                transform = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.RandomCrop(224),
                        transforms.RandomHorizontalFlip(),
                    ]
                )
            elif transform == "spatial_transform":
                transform = transforms.Compose(
                    [transforms.Resize(256), transforms.CenterCrop(224)]
                )
            else:
                print("Unknown transformation name")
                transform = None
        super(CUB2011, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.env = env
        self.train = train
        self.pred_feature = pred_feature
        self.prog_feature = prog_feature
        self.class_ids = class_ids
        self.loader = default_loader
        self.preload_imgs = preload_imgs
        self.load_1d_covariates = load_1d_covariates
        self.get_counterfactuals = get_counterfactuals
        self.x = None
        self.y_counterf = None
        self.treatidx_env_mapping = treatidx_env_mapping
        # swap values and keys for mapping
        self.env_treatidx_mapping = {v: k for k, v in treatidx_env_mapping.items()}

        attr_ids_dict = {
            "has_primary_color::black": 260,
            "b_colblack": 260,
            "has_bill_length::about_the_same_as_head": 150,
            "b_billsameashead": 150,
            "b_colwhite": 261,
            "has_primary_color::white": 261,
            "b_billlong": 151,
            "has_bill_length::longer_than_head": 151,
        }
        self.attr_ids = [
            attr_ids_dict[self.prog_feature],
            attr_ids_dict[self.pred_feature],
        ]
        # self.data_fun_name = data_fun_name
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted")

        # self._load_metadata()
        logger.info(
            f"Loading CUB_200_2011 dataset for environment {env} ({treatidx_env_mapping})."
        )
        self.target = self.load_target(**data_kwargs)

        if self.preload_imgs is not None:
            self.img_data = self.load_imgs()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.preload_imgs is None:
            sample = self.data.iloc[index]
            path = os.path.join(self.root, "images", sample.image_name)
            img = self.loader(path)
        else:
            img = self.img_data[index]

        target = self.target[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        if self.preload_imgs is None:
            for index, row in self.data.iterrows():
                filepath = os.path.join(self.root, "images", row.image_name)
                if not os.path.isfile(filepath):
                    logger.error(f"Image file not found: {filepath}")
                    return False
        return True

    def _load_metadata(self, val_split_size=0.2):
        suffix = "_allclasses" if self.class_ids is None else ""
        if not os.path.exists(
            os.path.join(self.root, "processed", f"metadata_{self.env}{suffix}.txt")
        ):
            logger.info(f"Loading and preprocessing metadata.")
            images = pd.read_csv(
                os.path.join(self.root, "images.txt"),
                sep=" ",
                names=["image_id", "image_name"],
            )

            image_attribute_labels = pd.read_csv(
                os.path.join(self.root, "attributes", "image_attribute_labels.txt"),
                sep=" ",
                on_bad_lines="warn",
                usecols=[0, 1, 2],
                names=["image_id", "attribute_id", "is_present"],
            )
            image_attribute_labels = image_attribute_labels[
                image_attribute_labels["attribute_id"].isin(self.attr_ids)
            ]
            image_attribute_labels = image_attribute_labels.pivot(
                index="image_id", columns="attribute_id", values="is_present"
            )

            train_test_split_df = pd.read_csv(
                os.path.join(self.root, "train_test_split.txt"),
                sep=" ",
                names=["image_id", "is_training_img"],
            )

            selected_idx = train_test_split_df.is_training_img == (self.env != "test")

            # get selected certain class_ids only
            if self.class_ids is not None:
                assert isinstance(self.class_ids, list)
                image_class_labels = pd.read_csv(
                    os.path.join(self.root, "image_class_labels.txt"),
                    sep=" ",
                    names=["image_id", "class_id"],
                )
                selected_idx = selected_idx & (
                    image_class_labels["class_id"].isin(self.class_ids)
                )

            df_data = images.merge(image_attribute_labels, on="image_id")
            df_data = df_data.merge(train_test_split_df, on="image_id")
            self.data = df_data[selected_idx.values]

            # if self.train:
            if self.env != "test" and ("traincv" not in self.env):
                data_train, data_val = train_test_split(
                    self.data, test_size=val_split_size, random_state=1
                )

                if self.env in ["CG", "EG", "train"]:
                    self.data = data_train
                elif self.env in ["CGval", "EGval", "val"]:
                    self.data = data_val

        else:
            filename = os.path.join(
                self.root, "processed", f"metadata_{self.env}{suffix}.txt"
            )
            logger.info(f"Getting metadata from text files: {filename}")

            attr_names = pd.read_csv(
                filename,
                sep=" ",
                nrows=0,
            ).columns.tolist()
            # convert attribute ids in column names to int
            attr_names = [int(x) if x.isdigit() else x for x in attr_names]
            self.data = pd.read_csv(filename, sep=" ", header=0, names=attr_names)

    def load_imgs(self):
        suffix = "_original" if self.preload_imgs == "original" else ""
        if self.class_ids is None:
            suffix += "_allclasses"
        filepath = os.path.join(self.root, "processed", f"imgs_{self.env}{suffix}.pt")
        logger.info(f"Loading CUB_200_2011 dataset from {filepath}.")
        if not os.path.isfile(filepath):
            logger.error(f"Image dataset {filepath} not found.")
        img_data = torch.load(filepath)
        return img_data

    def load_target(
        self, data_fun_name="linear", save_num_data_dir=None, **data_kwargs
    ):
        x = np.array(
            [self.data[self.attr_ids[0]].tolist(), self.data[self.attr_ids[1]].tolist()]
        ).T

        n_batch = len(self.data)
        selected_env = next(
            (v for k, v in self.env_treatidx_mapping.items() if self.env.startswith(k)),
            self.env,
        )
        data = get_data_fun[data_fun_name](
            env=selected_env,  # map treatment assigment
            n=n_batch,
            input_dim=3,
            x_input=torch.from_numpy(x[:n_batch]),
            custom_feature_idx=[0, 1],
            get_counterfactuals=self.get_counterfactuals,
            **data_kwargs,
        )

        (
            x,
            *target,
        ) = data
        if self.load_1d_covariates:
            self.x = x
        if self.get_counterfactuals:
            self.y_counterf = target[-1].squeeze(1)
            target = target[:-1]
        # target is either y or (y, treat) depending whether environment is specified
        if len(target) == 1:
            target = target[0].squeeze(1)
        elif len(target) == 2:
            target = torch.stack((target[0].squeeze(1), target[1]), dim=1)
        else:
            logger.warning(f"More than two targets received: {target}.")

        if save_num_data_dir is not None:  # saving numerical/tabular data only
            torch.save(data, os.path.join(save_num_data_dir, "numdata_train.pt"))
        return target


class ISIC2018(datasets.VisionDataset):
    """
    source: https://challenge.isic-archive.com/data/#2018

    Args:
        root (string): Root directory of dataset where ``img_*.pt`` is located
        env (string): Which environment to load, must start with 'CG' (control group) or 'EG' (experimental group), or be 'test'
        train (bool): Specifying training mode
        transform (Union[callable, str], optional): A string out of ["randomspatial_transform", "spatial_transform", "resize_randomspatial_transform","tensor_spatial_transform"] or function/transform that returns a transformed version of an image
        target_transform (callable, optional): A function/transform that takes returns a transformed version of a target
        pred_feature (str): Dataset attribute set as the predictive biomarker feature, e.g. "b_globules", "b_pignet"
        prog_feature (str): Dataset attribute set as the prognostic biomarker feature, e.g. "b_globules", "b_pignet"
        class_ids (list, optional): If specified: which classes from the dataset to include
        preload_imgs (str, optional): "original" for loading normalized tensor images, /not None/ for loading cropped images
        load_1d_covariates (bool, optional): Whether to also return the covariates used to simulate the target outcomes
        get_counterfactuals (bool, optional): Whether to also return the ground truth counterfactual outcomes with the target
        treatidx_env_mapping (dict, optional): A dictionary mapping treatment indices to environment labels, e.g., {0: "CG", 1: "EG"}
    """

    def __init__(
        self,
        root="/absolute/path/to/datasets/ISIC2018/",
        env="CG",
        train=True,
        transform=None,
        target_transform=None,
        pred_feature=None,
        prog_feature=None,
        class_ids=None,
        preload_imgs=None,
        load_1d_covariates=False,
        get_counterfactuals=False,
        treatidx_env_mapping={0: "CG", 1: "EG"},
        **data_kwargs,
    ):
        if (
            (env in ["test", "val", "CGval", "EGval"])
            or (env.endswith("val"))
            or (not train)
        ):
            logger.info(
                "Set default transform for test/validation data or if not in training mode."
            )
            transform = "spatial_transform"
            if preload_imgs is not None:
                preload_imgs = True  # use preprocessed/transformed data instead.
                logger.info("Using preprocessed data.")

        if (transform is not None) and isinstance(transform, str):
            if transform == "randomspatial_transform":
                transform = transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomApply(
                            [transforms.RandomRotation((90, 90))], p=0.5
                        ),
                        lambda x: to_tensor(x)
                        if not isinstance(x, torch.Tensor)
                        else x,
                        transforms.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04
                        ),
                        transforms.Normalize(
                            [0.7092, 0.5821, 0.5353], [0.1558, 0.1648, 0.1796]
                        ),
                    ]
                )
            elif transform == "spatial_transform":
                transform = transforms.Compose(
                    [
                        transforms.CenterCrop(224),
                        lambda x: to_tensor(x)
                        if not isinstance(x, torch.Tensor)
                        else x,
                        transforms.Normalize(
                            [0.7092, 0.5821, 0.5353], [0.1558, 0.1648, 0.1796]
                        ),
                    ]
                )
            elif transform == "resize_randomspatial_transform":
                transform = transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomApply(
                            [transforms.RandomRotation((90, 90))], p=0.5
                        ),
                        transforms.ColorJitter(
                            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04
                        ),
                        lambda x: to_tensor(x)
                        if not isinstance(x, torch.Tensor)
                        else x,
                        transforms.Normalize(
                            [0.7092, 0.5821, 0.5353], [0.1558, 0.1648, 0.1796]
                        ),
                    ]
                )
            elif transform == "tensor_spatial_transform":
                transform = transforms.Compose(
                    [
                        transforms.Resize(224),
                        transforms.CenterCrop(224),
                        # transforms.ToTensor(),
                        lambda x: to_tensor(x)
                        if not isinstance(x, torch.Tensor)
                        else x,
                        transforms.Normalize(
                            [0.7092, 0.5821, 0.5353], [0.1558, 0.1648, 0.1796]
                        ),
                    ]
                )
            else:
                print("Unknown transformation name")
                transform = None

        super(ISIC2018, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.env = env
        self.train = train
        self.pred_feature = pred_feature
        self.prog_feature = prog_feature
        self.class_ids = class_ids
        self.loader = Image.open  # default_loader
        self.preload_imgs = preload_imgs
        self.load_1d_covariates = load_1d_covariates
        self.get_counterfactuals = get_counterfactuals
        self.x = None
        self.y_counterf = None
        self.treatidx_env_mapping = treatidx_env_mapping
        self.env_treatidx_mapping = {v: k for k, v in treatidx_env_mapping.items()}
        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted")

        logger.info(
            f"Loading ISIC_2018 dataset for environment {env} ({treatidx_env_mapping})."
        )
        self.target = self.load_target(**data_kwargs)

        if self.preload_imgs is not None:
            self.img_data = self.load_imgs()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.preload_imgs is None:
            sample = self.data.iloc[index]
            path = os.path.join(self.root, "images", f"ISIC_{sample.img_id}.jpg")
            img = self.loader(path)
        else:
            img = self.img_data[index]

        target = self.target[index]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        if self.preload_imgs is None:
            for index, row in self.data.iterrows():
                filepath = os.path.join(self.root, "images", f"ISIC_{row.img_id}.jpg")
                if not os.path.isfile(filepath):
                    logger.error(f"Image file not found: {filepath}")
                    return False
        return True

    def _load_metadata(self, val_split_size=0.2):
        filename = os.path.join(self.root, "preprocessed", f"metadata_{self.env}.txt")
        logger.info(f"Getting metadata from text files: {filename}")

        self.data = pd.read_csv(filename, sep=" ", dtype={"img_id": object})

    def load_imgs(self):
        filepath = os.path.join(
            self.root, "preprocessed", f"imgs_{self.env}_resized.pt"
        )
        logger.info(f"Loading ISIC 2018 dataset from {filepath}.")
        if not os.path.isfile(filepath):
            logger.error(f"Image dataset {filepath} not found.")
        img_data = torch.load(filepath)
        return img_data

    def load_target(
        self, data_fun_name="linear", save_num_data_dir=None, **data_kwargs
    ):
        x = np.array(
            [
                self.data[self.prog_feature].tolist(),
                self.data[self.pred_feature].tolist(),
            ]
        ).T

        n_batch = len(self.data)

        # match env to the closes treatment assignment index
        selected_env = next(
            (v for k, v in self.env_treatidx_mapping.items() if self.env.startswith(k)),
            self.env,
        )

        data = get_data_fun[data_fun_name](
            env=selected_env,  # map treatment assigment
            n=n_batch,
            input_dim=3,
            x_input=torch.from_numpy(x[:n_batch]),
            custom_feature_idx=[0, 1],
            get_counterfactuals=self.get_counterfactuals,
            **data_kwargs,
        )

        (
            x,
            *target,
        ) = data
        # target is either y or (y, treat) depending whether environment is specified
        if self.load_1d_covariates:
            self.x = x
        if self.get_counterfactuals:
            self.y_counterf = target[-1].squeeze(1)
            target = target[:-1]
        if len(target) == 1:
            target = target[0].squeeze(1)
        elif len(target) == 2:
            target = torch.stack((target[0].squeeze(1), target[1]), dim=1)
        else:
            logger.warning(f"More than two targets received: {target}.")

        if save_num_data_dir is not None:  # saving numerical/tabular data only
            torch.save(data, os.path.join(save_num_data_dir, "numdata_train.pt"))
        return target


class NSCLCRadiomics(datasets.VisionDataset):
    """
    source: https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics#160568540f6d415ad92f4dbba0d89843bf5f1f14

    Args:
        root (string): Root directory of dataset for ``NSCLC_Radiomics``
        env (string): Which environment to load, must start with 'CG' (control group) or 'EG' (experimental group), or be 'test' or 'all_train'
        train (bool): Specifying training mode
        transform (string, callable, optional): A function/transform or string specifying a transform that takes in an image and returns a transformed version.
        target_transform (callable, optional): A function/transform or string specifying a transform that takes in the target and transforms it
        preload_imgs (string, optional): "original" for loading normalized tensor images, /not None/ for loading cropped images
        pred_feature (str): Dataset attribute set as the predictive biomarker feature, e.g. "Flatness_zscoretr", "Energy_zscoretr"
        prog_feature (str): Dataset attribute set as the prognostic biomarker feature, e.g. "Flatness_zscoretr", "Energy_zscoretr"
        load_1d_covariates (bool, optional): Whether to also return the covariates used to simulate the target outcomes
        get_counterfactuals (bool, optional): Whether to also return the ground truth counterfactual outcomes with the target
        transform_kwargs (dict, optional): Additional keyword arguments for the transform function
        preprocessed_folder_name (str, optional): Name of the preprocessed folder
        augmentation_framework (str, optional): Framework used for augmentation, e.g., "batchgenerators" or "monai"
        treatidx_env_mapping (dict, optional): A dictionary mapping treatment indices to environment labels, e.g., {0: "CG", 1: "EG"}
    """

    def __init__(
        self,
        root="/absolute/path/to/datasets/NSCLC_Radiomics",
        env="CG",
        train=True,
        transform=None,
        target_transform=None,
        pred_feature=None,
        prog_feature=None,
        load_1d_covariates=False,
        get_counterfactuals=False,
        transform_kwargs=None,
        preprocessed_folder_name="preprocessed",
        augmentation_framework="monai",
        treatidx_env_mapping={0: "CG", 1: "EG"},
        **data_kwargs,
    ):
        # set up transforms for 3d images for augmentation using batchgenerators
        if (transform is not None) and isinstance(transform, str):
            if transform_kwargs is None:
                transform_kwargs = {}
            if augmentation_framework == "batchgenerators":
                if not train:
                    # print("Set default transform for test/validation data.")
                    transform = "spatialpad_transform_CT"
                logger.info(
                    f"Using transform: {transform}, augmentation framework: {augmentation_framework}"
                )
                transform = get_batchgenerators_transforms(
                    augmentation_name=transform, **transform_kwargs
                )
            elif augmentation_framework == "monai":
                if not train:
                    # print("Set default transform for test/validation data.")
                    transform = "pad_transform_CT"
                logger.info(
                    f"Using transform: {transform}, augmentation framework: {augmentation_framework}"
                )
                transform = get_monai_transforms(
                    augmentation_name=transform, **transform_kwargs
                )
            else:
                logger.info(
                    f"Unknown augmentation framework: {augmentation_framework} for transform: {transform}"
                )
                print("Unknown augmentation framework")
                transform = None

        super(NSCLCRadiomics, self).__init__(
            root, transform=transform, target_transform=target_transform
        )

        self.env = env
        self.train = train
        self.prog_feature = prog_feature
        self.pred_feature = pred_feature
        # self.preload_imgs = preload_imgs
        self.load_1d_covariates = load_1d_covariates
        self.get_counterfactuals = get_counterfactuals
        self.data_kwargs = data_kwargs
        self.x = None
        self.y_counterf = None
        self.preprocessed_folder_name = preprocessed_folder_name
        self.augmentation_framework = augmentation_framework
        self.treatidx_env_mapping = treatidx_env_mapping
        self.env_treatidx_mapping = {v: k for k, v in treatidx_env_mapping.items()}

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. ")

        logger.info(
            f"Loading NSCLC Radiomics dataset for environment {env} ({treatidx_env_mapping})."
        )

        self.target = self.load_target(**data_kwargs)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        sample = self.data.iloc[index]
        path = os.path.join(
            self.root, self.preprocessed_folder_name, f"{sample.subject_id}.npy"
        )
        img = np.load(path, mmap_mode="r").astype(np.float32)
        target = self.target[index]

        if self.transform is not None:
            if self.augmentation_framework == "batchgenerators":
                img = self.transform(data=img[None, None, :])
                img = img["data"][0, :]  # additional dimension for batch size
            elif self.augmentation_framework == "monai":
                # ensure channel dimension is first (or use EnsureChannelFirstd)
                img = self.transform(img[None, :])

        if self.target_transform is not None:
            target = self.target_transform(data=target)

        return img, target

    def __len__(self):
        # return len(self.data_label_tuples)
        return len(self.data)

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(
                self.root, self.preprocessed_folder_name, f"{row.subject_id}.npy"
            )
            if not os.path.isfile(filepath):
                logger.error(f"Image file not found: {filepath}")
                return False
        return True

    def _load_metadata(self):
        filename = os.path.join(
            self.root, self.preprocessed_folder_name, f"metadata_{self.env}.txt"
        )
        logger.info(f"Getting metadata from text files: {filename}")
        self.data = pd.read_csv(
            filename,
            sep=" ",
        )

    def load_target(
        self, data_fun_name="linear", save_num_data_dir=None, **data_kwargs
    ):
        x = np.array(
            [
                self.data[self.prog_feature].tolist(),
                self.data[self.pred_feature].tolist(),
            ]
        ).T

        n_batch = len(self.data)
        if data_fun_name in ["linear", "logistic"]:
            selected_env = next(
                (
                    v
                    for k, v in self.env_treatidx_mapping.items()
                    if self.env.startswith(k)
                ),
                self.env,
            )
            data = get_data_fun[data_fun_name](
                env=selected_env,  # map treatment assigment
                n=n_batch,
                input_dim=3,
                x_input=torch.from_numpy(x[:n_batch]),
                custom_feature_idx=[0, 1],
                get_counterfactuals=self.get_counterfactuals,
                **data_kwargs,
            )

            (
                x,
                *target,
            ) = data

            # target is either y or (y, treat) depending whether environment is specified

        else:
            raise NotImplementedError
        if self.load_1d_covariates:
            self.x = x
        if self.get_counterfactuals:
            self.y_counterf = target[-1].squeeze(1)
            target = target[:-1]
        if len(target) == 1:
            target = target[0].squeeze(1)
        elif len(target) == 2:
            target = torch.stack((target[0].squeeze(1), target[1]), dim=1)
        else:
            logger.warning(f"More than two targets received: {target}.")

        if save_num_data_dir is not None:  # saving numerical/tabular data only
            torch.save(data, os.path.join(save_num_data_dir, "numdata_train.pt"))
        return target


class NSCLCRadiomicsSeg(datasets.VisionDataset):
    """
    source: https://wiki.cancerimagingarchive.net/display/Public/NSCLC-Radiomics#160568540f6d415ad92f4dbba0d89843bf5f1f14

    Args:
        root (string): Root directory of dataset for ``NSCLC_Radiomics``
        env (string): Which environment to load, must start with 'CG' (control group) or 'EG' (experimental group), or be 'test' or 'all_train'
        train (bool): Specifying training mode
        transform (string, callable, optional): A function/transform or string specifying a transform that takes in an image and returns a transformed version. E.g., ``transforms.RandomCrop``
        transform_kwargs (dict, optional): Additional keyword arguments for the transform function
        preprocessed_folder_name (str, optional): Name of the preprocessed folder
        augmentation_framework (str, optional): Framework used for augmentation, e.g., "batchgenerators" or "monai"
        treatidx_env_mapping (dict, optional): A dictionary mapping treatment indices to environment labels, e.g., {0: "CG", 1: "EG"}
    """

    def __init__(
        self,
        root="/absolute/path/to/datasets/NSCLC_Radiomics",
        env="CG",
        train=True,
        transform=None,
        transform_kwargs=None,
        preprocessed_folder_name="preprocessed_tumourbbox_patches_nnunetresample",
        augmentation_framework="monai",
        treatidx_env_mapping={0: "CG", 1: "EG"},
        **data_kwargs,
    ):
        # set up transforms for 3d images for augmentation using batchgenerators
        if (transform is not None) and isinstance(transform, str):
            if transform_kwargs is None:
                transform_kwargs = {}
            if augmentation_framework == "batchgenerators":
                transform = "spatialpad_transform_CT"
                transform = get_batchgenerators_transforms(
                    augmentation_name=transform, **transform_kwargs
                )
            elif augmentation_framework == "monai":
                transform = "pad_transform_CT"
                transform = get_monai_transforms(
                    augmentation_name=transform, **transform_kwargs
                )
            else:
                logger.info(
                    f"Unknown augmentation framework: {augmentation_framework} for transform: {transform}"
                )
                print("Unknown augmentation framework")
                transform = None

        super(NSCLCRadiomicsSeg, self).__init__(
            root,
            transform=transform,
        )

        self.env = env
        self.train = train
        self.data_kwargs = data_kwargs
        self.preprocessed_folder_name = preprocessed_folder_name
        self.augmentation_framework = augmentation_framework
        self.treatidx_env_mapping = treatidx_env_mapping
        self.env_treatidx_mapping = {v: k for k, v in treatidx_env_mapping.items()}

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. ")

        logger.info(
            f"Loading NSCLC Radiomics dataset for environment {env} ({treatidx_env_mapping})."
        )

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        sample = self.data.iloc[index]
        path = os.path.join(
            self.root,
            self.preprocessed_folder_name + "_seg",
            f"{sample.subject_id}_SEG.npy",
        )
        seg = np.load(path, mmap_mode="r").astype(np.uint8)
        if self.transform is not None:
            if self.augmentation_framework == "batchgenerators":
                seg = self.transform(data=seg[None, None, :])
                seg = seg["data"][0, :]  # additional dimension for batch size
            elif self.augmentation_framework == "monai":
                # ensure channel dimension is first (or use EnsureChannelFirstd)
                seg = self.transform(seg[None, :])
        return seg

    def __len__(self):
        # return len(self.data_label_tuples)
        return len(self.data)

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(
                self.root,
                self.preprocessed_folder_name + "_seg",
                f"{row.subject_id}_SEG.npy",
            )
            if not os.path.isfile(filepath):
                logger.error(f"Image file not found: {filepath}")
                return False
        return True

    def _load_metadata(self):
        filename = os.path.join(
            self.root, self.preprocessed_folder_name, f"metadata_{self.env}.txt"
        )
        logger.info(f"Getting metadata from text files: {filename}")
        self.data = pd.read_csv(
            filename,
            sep=" ",
        )


load_data_fn = {
    "colored_mnist": load_img_data_cmnist,
}

dataset_dict = {
    "colored_mnist": ColoredMNIST,
    "cub2011": CUB2011,
    "isic2018": ISIC2018,
    "nsclcradiomics": NSCLCRadiomics,
    "nsclcradiomicsseg": NSCLCRadiomicsSeg,
}

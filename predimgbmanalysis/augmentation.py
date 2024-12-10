import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torchvision import transforms

from sklearn.model_selection import train_test_split

import monai

from batchgenerators.transforms import (
    abstract_transforms,
    spatial_transforms,
    utility_transforms,
    crop_and_pad_transforms,
)


def get_torchvision_transforms(augmentation_name):
    if augmentation_name == "randomspatial_transform_cmnist":
        transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
            ]
        )
    elif augmentation_name == "resize_cmnist":
        transform = transforms.Compose([transforms.Resize((224, 224))])

    elif augmentation_name == "randomspatial_transform_cub2011":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        )
    elif augmentation_name == "spatial_transform_cub2011":
        transform = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224)]
        )

    elif augmentation_name == "randomspatial_transform_isic2018":
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04
                ),
                transforms.Normalize(
                    [0.7092, 0.5821, 0.5353], [0.1558, 0.1648, 0.1796]
                ),
            ]
        )
    elif augmentation_name == "spatial_transform_isic2018":
        transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    [0.7092, 0.5821, 0.5353], [0.1558, 0.1648, 0.1796]
                ),
            ]
        )
    elif augmentation_name == "resize_randomspatial_transform_isic2018":
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomResizedCrop(224, scale=(0.4, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.04
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.7092, 0.5821, 0.5353], [0.1558, 0.1648, 0.1796]
                ),
            ]
        )
    elif augmentation_name == "tensor_spatial_transform_isic2018":
        transform = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.7092, 0.5821, 0.5353], [0.1558, 0.1648, 0.1796]
                ),
            ]
        )
    else:
        print("Unknown transformation name")
        transform = None

    return transform


def get_monai_transforms(augmentation_name, **kwargs):
    all_transforms = []
    if "patch_size" in kwargs:
        patch_size = tuple(kwargs["patch_size"])
    else:
        patch_size = (224, 224, 112)
    all_transforms.append(monai.transforms.ToTensor())
    if augmentation_name == "randomspatialpad_transform_CT":
        all_transforms.append(
            monai.transforms.SpatialPad(
                patch_size, mode="constant", method="symmetric", constant_values=-1024
            )
        )

        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=0))
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=1))
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=2))
        all_transforms.append(
            monai.transforms.RandRotate90(prob=0.5, spatial_axes=(1, 2))
        )

    elif augmentation_name == "extendedrandomspatialpad_transform_CT":
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=0))
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=1))
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=2))
        all_transforms.append(
            monai.transforms.RandRotate90(prob=0.5, spatial_axes=(1, 2))
        )
        all_transforms.append(
            monai.transforms.SpatialPad(
                patch_size, mode="constant", method="symmetric", constant_values=-1024
            )
        )
        if "padding_mode" in kwargs:
            padding_mode = kwargs["padding_mode"]
        else:
            padding_mode = "constant"  # for backward compatibility, but "edge" or "minimum" is recommended
        all_transforms.append(
            monai.transforms.RandZoom(
                prob=0.5,
                min_zoom=0.9,
                max_zoom=1.1,
                keep_size=True,
                padding_mode=padding_mode,
                constant_values=-1024,
            )
        )

    elif augmentation_name == "randomspatialpad_rot_transform_CT":
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=0))
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=1))
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=2))
        all_transforms.append(
            monai.transforms.RandRotate90(prob=0.5, spatial_axes=(1, 2))
        )
        all_transforms.append(
            monai.transforms.SpatialPad(
                patch_size, mode="constant", method="symmetric", constant_values=-1024
            )
        )

        all_transforms.append(
            monai.transforms.RandRotate(
                prob=0.2,
                range_x=np.deg2rad(5),
                range_y=np.deg2rad(5),
                range_z=np.deg2rad(5),
                padding_mode="border",
                keep_size=True,
            )
        )

    elif augmentation_name == "randomspatialpad_rot_gaussian_transform_CT":
        NSCLCPATCH_STD = 361.7110462718799
        NSCLCPATCH_MEAN = -250.94877486254458
        print(f"Check Data Stats, using: {NSCLCPATCH_MEAN}, {NSCLCPATCH_STD}")
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=0))
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=1))
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=2))
        all_transforms.append(
            monai.transforms.RandRotate90(prob=0.5, spatial_axes=(1, 2))
        )
        all_transforms.append(
            monai.transforms.SpatialPad(
                patch_size, mode="constant", method="symmetric", constant_values=-1024
            )
        )
        # add gaussian noise
        all_transforms.append(
            monai.transforms.RandGaussianNoise(
                prob=0.15,
                mean=0.0,
                std=0.1 * NSCLCPATCH_STD,
            )
        )
        # add gaussian blur
        all_transforms.append(
            monai.transforms.RandGaussianSmooth(
                prob=0.1,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
            )
        )
        all_transforms.append(
            monai.transforms.RandRotate(
                prob=0.2,
                range_x=np.deg2rad(15),
                range_y=np.deg2rad(15),
                range_z=np.deg2rad(15),
                padding_mode="border",
                keep_size=True,
            )
        )

    elif augmentation_name == "randomspatialpad_rotfirst_gaussian_transform_CT":
        NSCLCPATCH_STD = 361.7110462718799
        NSCLCPATCH_MEAN = -250.94877486254458
        print(f"Check Data Stats, using: {NSCLCPATCH_MEAN}, {NSCLCPATCH_STD}")
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=0))
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=1))
        all_transforms.append(monai.transforms.RandFlip(prob=0.5, spatial_axis=2))
        all_transforms.append(
            monai.transforms.RandRotate90(prob=0.5, spatial_axes=(1, 2))
        )
        all_transforms.append(
            monai.transforms.SpatialPad(
                patch_size, mode="constant", method="symmetric", constant_values=-1024
            )
        )
        all_transforms.append(
            monai.transforms.RandRotate(
                prob=0.2,
                range_x=np.deg2rad(15),
                range_y=np.deg2rad(15),
                range_z=np.deg2rad(15),
                padding_mode="border",
                keep_size=True,
            )
        )
        # add gaussian noise
        all_transforms.append(
            monai.transforms.RandGaussianNoise(
                prob=0.15,
                mean=0.0,
                std=0.1 * NSCLCPATCH_STD,
            )
        )
        # add gaussian blur
        all_transforms.append(
            monai.transforms.RandGaussianSmooth(
                prob=0.1,
                sigma_x=(0.5, 1.5),
                sigma_y=(0.5, 1.5),
                sigma_z=(0.5, 1.5),
            )
        )

    elif augmentation_name == "pad_transform_CT":
        all_transforms.append(
            monai.transforms.SpatialPad(
                patch_size, mode="constant", method="symmetric", constant_values=-1024
            )
        )

    else:
        # no augmentation
        print("No augmentation")

    transform = monai.transforms.Compose(all_transforms)
    return transform

def get_batchgenerators_transforms(augmentation_name, **kwargs):
    all_transforms = []
    # check if patch_size is given as a parameter in kwargs and if so, use it
    if "patch_size" in kwargs:
        patch_size = tuple(kwargs["patch_size"])
    else:
        patch_size = (224, 224, 112)

    if augmentation_name == "randomspatial_transform":
        all_transforms.append(
            spatial_transforms.SpatialTransform(
                patch_size=patch_size,  # crop z axis to mean z-size of all images
                patch_center_dist_from_border=tuple(
                    int(element / 2) for element in patch_size
                ),  # patch size / 2
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=(-5.0 / 360 * 2.0 * np.pi, 5.0 / 360 * 2.0 * np.pi),
                angle_y=(-5.0 / 360 * 2.0 * np.pi, 5.0 / 360 * 2.0 * np.pi),
                angle_z=(-5.0 / 360 * 2.0 * np.pi, 5.0 / 360 * 2.0 * np.pi),
                do_scale=True,
                scale=(0.9, 1.1),
                border_mode_data="constant",
                border_cval_data=0,  
                order_data=3,
                random_crop=True,
                p_el_per_sample=0.2,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.2,
            )
        )
        # mirror
        all_transforms.append(spatial_transforms.MirrorTransform(axes=(0, 1, 2)))

    elif augmentation_name == "spatial_transform":
        all_transforms.append(
            spatial_transforms.SpatialTransform(
                patch_size=patch_size,  # crop z axis to mean z-size of all images
                do_elastic_deform=False,
                do_rotation=False,
                do_scale=False,
                border_mode_data="constant",
                border_cval_data=0,  
                order_data=3,
                random_crop=False,
                p_el_per_sample=0.0,
                p_scale_per_sample=0,
                p_rot_per_sample=0.0,
            )
        )

    elif augmentation_name == "randomspatialpad_transform_CT":
        all_transforms.append(
            crop_and_pad_transforms.PadTransform(
                new_size=patch_size,
                pad_mode_data="constant",
                np_pad_kwargs_data={"constant_values": -1024},
            )
        )
        all_transforms.append(
            spatial_transforms.SpatialTransform(
                patch_size=patch_size,  # crop z axis to mean z-size of all images
                patch_center_dist_from_border=tuple(
                    int(element / 2) for element in patch_size
                ),  # patch size / 2
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=(-5.0 / 360 * 2.0 * np.pi, 5.0 / 360 * 2.0 * np.pi),
                angle_y=(-5.0 / 360 * 2.0 * np.pi, 5.0 / 360 * 2.0 * np.pi),
                angle_z=(-5.0 / 360 * 2.0 * np.pi, 5.0 / 360 * 2.0 * np.pi),
                do_scale=True,
                scale=(0.9, 1.1),
                border_mode_data="constant",
                border_cval_data=-1024, 
                order_data=3,
                random_crop=True,
                p_el_per_sample=0.2,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.2,
            )
        )
        all_transforms.append(spatial_transforms.MirrorTransform(axes=(0, 1, 2)))

    elif augmentation_name == "spatialpad_transform_CT":
        all_transforms.append(
            crop_and_pad_transforms.PadTransform(
                new_size=patch_size,
                pad_mode_data="constant",
                np_pad_kwargs_data={"constant_values": -1024},
            )
        )
        all_transforms.append(
            spatial_transforms.SpatialTransform(
                patch_size=patch_size,  # crop z axis to mean z-size of all images
                do_elastic_deform=False,
                do_rotation=False,
                do_scale=False,
                border_mode_data="constant",
                border_cval_data=-1024,  
                order_data=3,
                random_crop=False,
                p_el_per_sample=0.0,
                p_scale_per_sample=0,
                p_rot_per_sample=0.0,
            )
        )

    elif augmentation_name == "pad_transform_CT":
        all_transforms.append(
            crop_and_pad_transforms.PadTransform(
                new_size=patch_size,
                pad_mode_data="constant",
                np_pad_kwargs_data={"constant_values": -1024.0},
            )
        )

    elif augmentation_name == "randomspatial_transform_custom":
        all_transforms.append(spatial_transforms.SpatialTransform(**kwargs))
        all_transforms.append(spatial_transforms.MirrorTransform(axes=(0, 1, 2)))

    elif augmentation_name == "simple_transform":
        all_transforms.append(
            crop_and_pad_transforms.CenterCropTransform(crop_size=(224, 224, 112))
        )
    else:
        print("No augmentation")

    all_transforms.append(utility_transforms.NumpyToTensor())
    # print("Transforms: ", all_transforms)
    transform = abstract_transforms.Compose(all_transforms)

    return transform
import os
import pandas as pd

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torchvision.datasets.folder import default_loader

from sklearn.model_selection import train_test_split
from skimage import io, transform
import cv2

from tqdm import tqdm
import tqdm.notebook as tq

import pydicom
import pydicom_seg
from pydicom import dcmread
from pydicom.data import get_testdata_file
import SimpleITK as sitk

from radiomics import (
    featureextractor,
)  # This module is used for interaction with pyradiomics

import json


def resampling_data(
    image,
    resampling_mode,
    medianspacing,
    anisotropy_threshold,
    resize=False,
    is_seg=False,
    image_metadata=None,
):
    if resampling_mode == "nnunet":
        try:
            from nnunetv2.preprocessing.resampling.default_resampling import (
                resample_data_or_seg_to_spacing,
            )
        except ImportError:
            print("Error: nnunetv2 is required for this preprocessing step.")
            print("Please install nnunetv2 using the following command:")
            print("pip install nnunetv2")
        image_array = sitk.GetArrayFromImage(image)
        image_array = image_array.transpose(2, 1, 0)
        current_spacing = np.array(image.GetSpacing())
        # add channel dimension so data is (c,x,y,z)
        image_array = np.expand_dims(image_array, axis=0)
        # check nnunet function!
        image_array = resample_data_or_seg_to_spacing(
            image_array,
            current_spacing=current_spacing,
            new_spacing=medianspacing,
            is_seg=is_seg,
        )
        image = sitk.GetImageFromArray(image_array[0].transpose(2, 1, 0))
        image_array = image_array[0].transpose(2, 1, 0)
        image.SetSpacing(medianspacing)
        image.SetOrigin(image.GetOrigin())
        image.SetDirection(image.GetDirection())
    elif resampling_mode == "nnunet_numpyarray":
        image_array = image
        current_spacing = np.array(image_metadata["original_spacing"])
        image_array = resample_data_or_seg_to_spacing(
            image_array,
            current_spacing=current_spacing,
            new_spacing=medianspacing,
            is_seg=is_seg,
        )

    else:
        # using sitk resampler
        if is_seg:
            seg_array = sitk.GetArrayFromImage(image)
            seg_array_onehot = np.zeros(
                (seg_array.shape[0], seg_array.shape[1], seg_array.shape[2], 2)
            )
            seg_array_onehot[:, :, :, 0] = seg_array == 0
            seg_array_onehot[:, :, :, 1] = seg_array > 0
            seg_onehot = sitk.GetImageFromArray(seg_array_onehot)
            seg_onehot.CopyInformation(image)
            # create resampler for one-hot encoded segmentation
            resampler_seg_onehot = sitk.ResampleImageFilter()
            resampler_seg_onehot.SetOutputDirection(seg_onehot.GetDirection())
            resampler_seg_onehot.SetOutputOrigin(seg_onehot.GetOrigin())
            if (
                np.max(medianspacing) / np.min(medianspacing)
            ) > anisotropy_threshold and resampling_mode != "sitk_isotropic":
                # use nearest neighbor interpolation for anisotropic spacing in z direction and linear interpolation for isotropic spacing
                # get which dimension is anisotropic
                anisotropic_dim = np.argmax(medianspacing)

                # resample in isotropic dimensions using linear interpolation
                resampler_seg_onehot.SetInterpolator(sitk.sitkLinear)

                # get original spacing
                originalspacing = np.array(image.GetSpacing())
                medianspacing_isotropic = medianspacing.copy()
                medianspacing_isotropic[anisotropic_dim] = originalspacing[
                    anisotropic_dim
                ]
                resampler_seg_onehot.SetOutputSpacing(medianspacing_isotropic)
                resampler_seg_onehot.SetSize(seg_onehot.GetSize())
                seg_onehot = resampler_seg_onehot.Execute(seg_onehot)
                # resample in anisotropic dimension using nearest neighbor interpolation
                resampler_seg_onehot.SetOutputDirection(seg_onehot.GetDirection())
                resampler_seg_onehot.SetOutputOrigin(seg_onehot.GetOrigin())
                resampler_seg_onehot.SetInterpolator(sitk.sitkNearestNeighbor)
                resampler_seg_onehot.SetOutputSpacing(medianspacing)
                resampler_seg_onehot.SetSize(seg_onehot.GetSize())
                seg_onehot = resampler_seg_onehot.Execute(seg_onehot)

            else:
                # use linear interpolation for isotropic spacing
                resampler_seg_onehot.SetInterpolator(sitk.sitkLinear)

                resampler_seg_onehot.SetOutputSpacing(medianspacing)
                resampler_seg_onehot.SetSize(seg_onehot.GetSize())

                seg_onehot = resampler_seg_onehot.Execute(seg_onehot)

            seg_array_onehot = sitk.GetArrayFromImage(
                seg_onehot
            )  # dimension of seg_array_onehot is (z,y,x,2)
            # convert seg_array_onehot to one channel segmentation using argmax
            seg_array = np.argmax(seg_array_onehot, axis=-1)
            image = seg_onehot
            image_array = seg_array
        else:
            resampler = sitk.ResampleImageFilter()
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetOutputOrigin(image.GetOrigin())
            if (
                np.max(medianspacing) / np.min(medianspacing)
            ) > anisotropy_threshold and resampling_mode != "sitk_isotropic":
                # get which dimension is anisotropic
                anisotropic_dim = np.argmax(medianspacing)
                # resample in isotropic dimensions using bspline interpolation
                resampler.SetInterpolator(sitk.sitkBSpline)
                originalspacing = np.array(image.GetSpacing())
                medianspacing_isotropic = medianspacing.copy()
                medianspacing_isotropic[anisotropic_dim] = originalspacing[
                    anisotropic_dim
                ]
                if resize:
                    resampler.SetOutputSpacing(
                        [
                            medianspacing_isotropic[0] * 2,
                            medianspacing_isotropic[1] * 2,
                            medianspacing_isotropic[2],
                        ]
                    )
                    resampler.SetSize(
                        [
                            int(image.GetSize()[0] / 2),
                            int(image.GetSize()[1] / 2),
                            image.GetSize()[2],
                        ]
                    )
                else:
                    resampler.SetOutputSpacing(medianspacing_isotropic)
                    resampler.SetSize(image.GetSize())
                image = resampler.Execute(image)

                # resample in anisotropic dimension using nearest neighbor interpolation
                resampler.SetOutputDirection(image.GetDirection())
                resampler.SetOutputOrigin(image.GetOrigin())
                resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                if resize:
                    resampler.SetOutputSpacing(
                        [medianspacing[0] * 2, medianspacing[1] * 2, medianspacing[2]]
                    )
                    resampler.SetSize(
                        [
                            int(image.GetSize()[0] / 2),
                            int(image.GetSize()[1] / 2),
                            image.GetSize()[2],
                        ]
                    )
                else:
                    resampler.SetOutputSpacing(medianspacing)
                    resampler.SetSize(image.GetSize())
                image = resampler.Execute(image)
            else:
                # use the same interpolator for all dimensions
                resampler.SetInterpolator(sitk.sitkBSpline)
                # adjust medianspacing so that the image size is decreased by a factor of 2 in the x and y direction
                if resize:
                    resampler.SetOutputSpacing(
                        [medianspacing[0] * 2, medianspacing[1] * 2, medianspacing[2]]
                    )

                    resampler.SetSize(
                        [
                            int(image.GetSize()[0] / 2),
                            int(image.GetSize()[1] / 2),
                            image.GetSize()[2],
                        ]
                    )
                else:
                    resampler.SetOutputSpacing(medianspacing)
                    resampler.SetSize(image.GetSize())

                image = resampler.Execute(image)
                image_array = sitk.GetArrayFromImage(
                    image
                )  # dimension of image_array is (z,y,x)

    return image, image_array


def preprocess_img_data_cub2011(
    # env,
    root="/absolute/path/to/datasets/CUB_200_2011/CUB_200_2011/",
    class_ids=None,
    features=["b_colwhite", "b_billlong"],
    val_split_size=0.2,
    resize=True,
):
    assert features[0] != features[1]
    assert (val_split_size >= 0) and (
        val_split_size <= 1
    ), "val_split size must be in range [0,1]"

    resize_suffix = "" if resize else "_original"
    classid_suffix = "_allclasses" if class_ids is None else ""
    ds_means = (
        [0.4851, 0.4990, 0.4326] if class_ids is None else [0.4493, 0.4655, 0.4061]
    )
    ds_stds = (
        [0.2311, 0.2269, 0.2649] if class_ids is None else [0.2407, 0.2389, 0.2708]
    )
    if resize:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(ds_means, ds_stds),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(ds_means, ds_stds),
            ]
        )
    if not os.path.exists(os.path.join(root, "processed")):
        os.mkdir(os.path.join(root, "processed"))

    suffix_list = ["CG", "EG", "CGval", "EGval", "val", "test"]

    if not all(
        [
            os.path.isfile(
                os.path.join(
                    root, "processed", f"metadata_{suffix}{classid_suffix}.txt"
                )
            )
            for suffix in suffix_list
        ]
    ):
        attr_ids_dict = {
            "b_colwhite": 261,
            "has_primary_color::white": 261,
            "b_billlong": 151,
            "has_bill_length::longer_than_head": 151,
        }
        attr_ids = [attr_ids_dict[f] for f in features]

        images = pd.read_csv(
            os.path.join(root, "images.txt"), sep=" ", names=["image_id", "image_name"]
        )
        image_attribute_labels = pd.read_csv(
            os.path.join(root, "attributes", "image_attribute_labels.txt"),
            sep=" ",
            on_bad_lines="warn",
            usecols=[0, 1, 2],
            names=["image_id", "attribute_id", "is_present"],
        )

        image_attribute_labels = image_attribute_labels[
            image_attribute_labels["attribute_id"].isin(attr_ids)
        ]
        image_attribute_labels = image_attribute_labels.pivot(
            index="image_id", columns="attribute_id", values="is_present"
        )

        train_test_split_df = pd.read_csv(
            os.path.join(root, "train_test_split.txt"),
            sep=" ",
            names=["image_id", "is_training_img"],
        )
        df_data = images.merge(image_attribute_labels, on="image_id")
        df_data = df_data.merge(train_test_split_df, on="image_id")

        data_list = []
        for train in [1, 0]:
            selected_idx = df_data.is_training_img == train

            # get selected certain class_ids only
            if class_ids is not None:
                assert isinstance(class_ids, list)
                image_class_labels = pd.read_csv(
                    os.path.join(root, "image_class_labels.txt"),
                    sep=" ",
                    names=["image_id", "class_id"],
                )
                selected_idx = selected_idx & (
                    image_class_labels["class_id"].isin(class_ids)
                )

            data = df_data[selected_idx.values]
            if train:
                # train validation split
                data_splits = train_test_split(
                    data, test_size=val_split_size, random_state=1
                )

                for data_split, suffix in zip(data_splits, ["", "val"]):
                    data_split_envs = train_test_split(
                        data_split, test_size=0.5, random_state=1
                    )
                    print(f"saving metadata_*{suffix}.txt")
                    data_split_envs[0].to_csv(
                        os.path.join(
                            root,
                            "processed",
                            f"metadata_CG{suffix}{classid_suffix}.txt",
                        ),
                        index=False,
                        sep=" ",
                    )
                    data_split_envs[1].to_csv(
                        os.path.join(
                            root,
                            "processed",
                            f"metadata_EG{suffix}{classid_suffix}.txt",
                        ),
                        index=False,
                        sep=" ",
                    )
                    data_list.extend(data_split_envs)
                    print(f"saving metadata_val.txt")
                    data_splits[1].to_csv(
                        os.path.join(
                            root, "processed", f"metadata_val{classid_suffix}.txt"
                        ),
                        index=False,
                        sep=" ",
                    )
                data_list.append(data_splits[1])
            else:
                data.to_csv(
                    os.path.join(
                        root, "processed", f"metadata_test{classid_suffix}.txt"
                    ),
                    index=False,
                    sep=" ",
                )
                print(f"saving metadata_test.txt")
                data_list.append(data)

    else:
        print("Load existing *_metadata.txt files.")
        data_list = []
        for suffix in suffix_list:
            data_list.append(
                pd.read_csv(
                    os.path.join(
                        root, "processed", f"metadata_{suffix}{classid_suffix}.txt"
                    ),
                    sep=" ",
                )
            )
    print([len(d_) for d_ in data_list])
    if not all(
        [
            os.path.isfile(
                os.path.join(
                    root,
                    "processed",
                    f"imgs_{suffix}{resize_suffix}{classid_suffix}.pt",
                )
            )
            for suffix in suffix_list
        ]
    ):
        # save resized and cropped images
        for data, suffix in zip(data_list, suffix_list):
            imgs = []
            save_path = os.path.join(
                root,
                "processed",
                f"imgs_{suffix}{resize_suffix}{classid_suffix}.pt",
            )
            if not os.path.isfile(save_path):
                for i in range(len(data)):
                    path = os.path.join(root, f"images", data.iloc[i]["image_name"])

                    imgs.append(transform(default_loader(path)))

                if resize:
                    torch.save(torch.stack(imgs), save_path)
                else:
                    torch.save(imgs, save_path)
                print(f"saving {save_path}")

    else:
        print("*_imgs.pt files already exist.")


def attributes_to_df_isic2018(
    folder,
    root="/absolute/path/to/datasets/ISIC2018/",
    attribute_names=None,
    n_total=None,
):
    save_path = os.path.join(root, "preprocessed", f"attributes_{folder}.txt")
    if not os.path.isfile(save_path):
        img_ids = []
        attribute_abbr = ["globules", "mlc", "negnet", "pignet", "streaks"]
        if attribute_names is None:
            attribute_names = [
                "globules",
                "milia_like_cyst",
                "negative_network",
                "pigment_network",
                "streaks",
            ]
        for filename in sorted(os.listdir(os.path.join(root, folder))):
            if filename.split(".")[-1] == "png":
                img_id = filename.split("_")[1]
                img_ids.append(img_id)
        img_ids = list(set(img_ids))
        if n_total is None or n_total > len(img_ids):
            n_total = len(img_ids)

        df = pd.DataFrame(img_ids[:n_total], columns=["img_id"])
        for i, attr in enumerate(tq.tqdm(attribute_names)):
            attr_rel = []
            attr_abs = []
            b_attr = []
            for id_ in tq.tqdm(img_ids[:n_total]):
                img = cv2.imread(
                    os.path.join(root, folder, f"ISIC_{id_}_attribute_{attr}.png")
                )
                n_pix = img.sum()
                attr_rel.append(n_pix / (3 * 255 * np.prod(img.shape)))
                attr_abs.append(n_pix)
                b_attr.append(int(n_pix > 0))
            df[f"c_rel_{attribute_abbr[i]}"] = attr_rel
            df[f"c_{attribute_abbr[i]}"] = attr_abs
            df[f"b_{attribute_abbr[i]}"] = b_attr

        print(f"saving {save_path}")
        df.to_csv(save_path, index=False, sep=" ")
    else:
        print(f"attributes_{folder}.txt already exists.")
        df = pd.read_csv(save_path, sep=" ")
    return df


def preprocess_img_data_isic2018_trainvaltest(
    attr_filename,
    root="/absolute/path/to/datasets/ISIC2018/",
    val_test_split_size=[0.1, 0.2],
    resize=True,
    preprocess_imgs=False,
):
    assert (sum(val_test_split_size) >= 0) and (
        sum(val_test_split_size) <= 1
    ), "val_test_split size must be in range [0,1]"

    resize_suffix = "" if resize else "_original"
    ds_means = [0.7070, 0.5805, 0.5348]
    ds_stds = [0.1560, 0.1665, 0.1824]
    if resize:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(ds_means, ds_stds),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(ds_means, ds_stds),
            ]
        )
    if not os.path.exists(os.path.join(root, "preprocessed")):
        os.mkdir(os.path.join(root, "preprocessed"))

    suffix_list = ["CG", "EG", "CGval", "EGval", "test"]

    if not all(
        [
            os.path.isfile(os.path.join(root, "preprocessed", f"metadata_{suffix}.txt"))
            for suffix in suffix_list
        ]
    ):
        df_data = pd.read_csv(
            os.path.join(root, "preprocessed", attr_filename),
            sep=" ",
            dtype={"img_id": object},
        )

        data_list = []

        # train validation split
        data_splits_trainval, data_split_test = train_test_split(
            df_data, test_size=val_test_split_size[1], random_state=1
        )

        # train and val
        data_splits = train_test_split(
            data_splits_trainval,
            test_size=val_test_split_size[0] / (1.0 - val_test_split_size[1]),
            random_state=1,
        )
        print(len(data_splits[0]))
        for data_split, suffix in zip(data_splits, ["", "val"]):
            data_split_envs = train_test_split(
                data_split, test_size=0.5, random_state=1
            )

            print(f"saving metadata_*{suffix}.txt")
            data_split_envs[0].to_csv(
                os.path.join(root, "preprocessed", f"metadata_CG{suffix}.txt"),
                index=False,
                sep=" ",
            )
            data_split_envs[1].to_csv(
                os.path.join(root, "preprocessed", f"metadata_EG{suffix}.txt"),
                index=False,
                sep=" ",
            )
            data_list.extend(data_split_envs)

        # test
        data_split_test.to_csv(
            os.path.join(root, "preprocessed", f"metadata_test.txt"),
            index=False,
            sep=" ",
        )
        print(f"saving metadata_test.txt")
        data_list.append(data_split_test)

    else:
        print("Load existing *_metadata.txt files.")
        data_list = []
        for suffix in suffix_list:
            data_list.append(
                pd.read_csv(
                    os.path.join(root, "preprocessed", f"metadata_{suffix}.txt"),
                    sep=" ",
                    dtype={"img_id": object},
                )
            )

    print([len(d_) for d_ in data_list])
    if preprocess_imgs:
        if not all(
            [
                os.path.isfile(
                    os.path.join(
                        root, "preprocessed", f"imgs_{suffix}{resize_suffix}.pt"
                    )
                )
                for suffix in suffix_list
            ]
        ):
            # save resized and cropped images
            for data, suffix in zip(data_list, suffix_list):
                imgs = []
                for i in range(len(data)):
                    path = os.path.join(
                        root, "images", f"ISIC_{data.iloc[i]['img_id']}.jpg"
                    )
                    imgs.append(transform(default_loader(path)))
                    save_path = os.path.join(
                        root, "preprocessed", f"imgs_{suffix}{resize_suffix}.pt"
                    )
                if resize:
                    torch.save(torch.stack(imgs), save_path)
                else:
                    torch.save(imgs, save_path)
                print(f"saving {save_path}")
                break

        else:
            print("*_imgs.pt files already exist.")


def preprocess_img_data_isic2018(
    attr_filename,
    root="/absolute/path/to/datasets/ISIC2018/",
    val_split_size=0.2,
    get_trainval_split=False,
    preprocess_imgs=False,
):
    assert (val_split_size >= 0) and (
        val_split_size <= 1
    ), "val_test_split size must be in range [0,1]"

    if not os.path.exists(os.path.join(root, "preprocessed")):
        os.mkdir(os.path.join(root, "preprocessed"))

    if get_trainval_split:
        suffix_list = ["CG", "EG", "CGval", "EGval", "val"]
    else:
        suffix_list = ["test"]
    # suffix_list = ["CG", "EG", "train", "CGval", "EGval", "test"]
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.7092, 0.5821, 0.5353], [0.1558, 0.1648, 0.1796]),
        ]
    )

    if not all(
        [
            os.path.isfile(os.path.join(root, "preprocessed", f"metadata_{suffix}.txt"))
            for suffix in suffix_list
        ]
    ):
        df_data = pd.read_csv(
            os.path.join(root, "preprocessed", attr_filename),
            sep=" ",
            dtype={"img_id": object},
        )

        data_list = []
        if get_trainval_split:
            # train and val
            data_splits = train_test_split(
                df_data, test_size=val_split_size, random_state=1
            )
            data_splits[0].to_csv(
                os.path.join(root, "preprocessed", f"metadata_train.txt"),
                index=False,
                sep=" ",
            )
            data_splits[1].to_csv(
                os.path.join(root, "preprocessed", f"metadata_val.txt"),
                index=False,
                sep=" ",
            )
            for data_split, suffix in zip(data_splits, ["", "val"]):
                data_split_envs = train_test_split(
                    data_split, test_size=0.5, random_state=1
                )

                print(f"saving metadata_*{suffix}.txt")
                data_split_envs[0].to_csv(
                    os.path.join(root, "preprocessed", f"metadata_CG{suffix}.txt"),
                    index=False,
                    sep=" ",
                )
                print(len(data_split_envs[0]))
                data_split_envs[1].to_csv(
                    os.path.join(root, "preprocessed", f"metadata_EG{suffix}.txt"),
                    index=False,
                    sep=" ",
                )
                print(len(data_split_envs[1]))
                data_list.extend(data_split_envs)
        else:
            # test
            df_data.to_csv(
                os.path.join(root, "preprocessed", f"metadata_test.txt"),
                index=False,
                sep=" ",
            )
            print(len(df_data))
            print(f"saving metadata_test.txt")
            data_list.append(df_data)

    else:
        print("*_metadata.txt files already exits.")
        data_list = []
        for suffix in suffix_list:
            data_list.append(
                pd.read_csv(
                    os.path.join(root, "preprocessed", f"metadata_{suffix}.txt"),
                    sep=" ",
                    dtype={"img_id": object},
                )
            )

    print([len(d_) for d_ in data_list])
    if preprocess_imgs:
        if not all(
            [
                os.path.isfile(
                    os.path.join(root, "preprocessed", f"imgs_{suffix}_resized.pt")
                )
                for suffix in suffix_list
            ]
        ):
            # save resized and cropped images

            for data, suffix in zip(data_list, suffix_list):
                save_path = os.path.join(
                    root, "preprocessed", f"imgs_{suffix}_resized.pt"
                )
                if not os.path.isfile(save_path):
                    print(f"preprocessing {suffix} imgs")
                    imgs = []
                    for i in tqdm(range(len(data))):
                        path = os.path.join(
                            root, "images", f"ISIC_{data.iloc[i]['img_id']}.jpg"
                        )
                        imgs.append(transform(default_loader(path)))

                    torch.save(imgs, save_path)

                    print(f"saving {save_path}")
                else:
                    print(f"*_imgs_{suffix}.pt files already exist.")

        else:
            print("*_imgs.pt files already exist.")


def preprocess_nsclc_radiomics(
    folder="NSCLC-Radiomics",
    root="/absolute/path/to/datasets/NSCLC_Radiomics/",
    metadata_filename="metadata.csv",
    clinicaldata_filename="NSCLC Radiomics Lung1.clinical-version3-Oct 2019.csv",
    medianspacing=[0.9765625, 0.9765625, 3.0],
    resize=True,
    get_largest_tumour_component_only=False,
    get_tumour_bbox_patches=False,
    tumour_crop_margin=[2, 6, 6],
    anisotropy_threshold=3.0,
    resampling_mode="sitk_isotropic",
    preprocessed_folder_name=None,
    normalize_intensity=False,
    global_mean=-250.94877486254458,
    global_std=361.7110462718799,
    global_5p=-845.4839016685603,
    global_95p=158.39093935142375,
):
    if preprocessed_folder_name is None:
        if get_tumour_bbox_patches:
            preprocessed_folder_name = "preprocessed_tumourbbox_patches"
        else:
            preprocessed_folder_name = "preprocessed"

    if not os.path.exists(os.path.join(root, preprocessed_folder_name)):
        os.makedirs(os.path.join(root, preprocessed_folder_name))

    # save preprocessing parameters as json file
    with open(
        os.path.join(root, preprocessed_folder_name, f"preprocessing_parameters.json"),
        "w",
    ) as f:
        json.dump(
            {
                "medianspacing": medianspacing,
                "resize": resize,
                "get_largest_tumour_component_only": get_largest_tumour_component_only,
                "get_tumour_bbox_patches": get_tumour_bbox_patches,
                "tumour_crop_margin": tumour_crop_margin,
                "anisotropy_threshold": anisotropy_threshold,
                "resampling_mode": resampling_mode,
                "normalize_intensity": normalize_intensity,
                "global_mean": global_mean,
                "global_std": global_std,
                "global_5p": global_5p,
                "global_95p": global_95p,
            },
            f,
            indent=4,
        )

    metadata = pd.read_csv(
        os.path.join(root, metadata_filename),
    )

    clinicaldata = pd.read_csv(
        os.path.join(root, clinicaldata_filename),
    )

    data = []
    for i, row in tqdm(
        metadata[metadata["Modality"] == "CT"].iterrows(),
        total=metadata[metadata["Modality"] == "CT"].shape[0],
    ):
        subject_id = row["Subject ID"]
        if subject_id not in [
            "LUNG1-014",
            "LUNG1-021",
            "LUNG1-085",
            "LUNG1-128",
        ]:  # filter out ids with missing slices
            filename = os.path.join(root, os.path.normpath(row["File Location"]))
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(filename)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()  # dimensions are (x,y,z)

            segrow = metadata[
                (metadata["Modality"] == "SEG") & (metadata["Subject ID"] == subject_id)
            ]
            if segrow.shape[0] > 1:
                print(
                    f"Warning: More than one segmentation found for Subject ID = {subject_id}: {segrow.shape[0]}"
                )
            segrow = segrow.iloc[0]
            segfilename = os.path.join(
                root, os.path.normpath(segrow["File Location"]), "1-1.dcm"
            )

            dcm = pydicom.dcmread(segfilename)
            reader = pydicom_seg.SegmentReader()
            result = reader.read(dcm)

            tumor_labels = []
            lung_labels = []
            for segment_number in result.available_segments:
                if "GTV" in result.segment_infos[segment_number].SegmentDescription:
                    tumor_labels.append(segment_number)
                if "Lung" in result.segment_infos[segment_number].SegmentDescription:
                    lung_labels.append(segment_number)

            if len(tumor_labels) > 1:
                print(
                    f"Warning: More than one tumor segmentation found for Subject ID = {subject_id}: {len(tumor_labels)}"
                )

            segment_number = tumor_labels[0]
            seg = result.segment_image(segment_number)  # lazy construction
            # skip if image and segmentation have different dimensions
            if image.GetSize() != seg.GetSize():
                print(
                    f"Warning: Image and segmentation have different dimensions for Subject ID = {subject_id}. Skipping..."
                )
                continue

            if get_largest_tumour_component_only:
                # get largest tumour connected component only with sitk
                cc_seg = sitk.ConnectedComponent(seg)
                sorted_cc_seg = sitk.RelabelComponent(cc_seg, sortByObjectSize=True)
                seg = sitk.Cast(sorted_cc_seg == 1, sitk.sitkUInt8)

            if normalize_intensity:
                # clip intensity values to 5th and 95th percentile
                image_temp = image
                image_array = sitk.GetArrayFromImage(image)
                image_array = np.clip(image_array, global_5p, global_95p)
                # subtract global mean and divide by global std
                image_array = (image_array - global_mean) / global_std
                image = sitk.GetImageFromArray(image_array)
                image.CopyInformation(image_temp)
                image.SetSpacing(image_temp.GetSpacing())
                image.SetOrigin(image_temp.GetOrigin())
                image.SetDirection(image_temp.GetDirection())

            # get radiomics features
            extractor = featureextractor.RadiomicsFeatureExtractor(
                geometryTolerance=1e-4
            )
            extractor.disableAllFeatures()
            extractor.disableAllImageTypes()
            extractor.enableImageTypeByName("Original")
            extractor.enableImageTypeByName("Wavelet")
            # extractor.enableFeaturesByName(firstorder=['Energy','TotalEnergy','Entropy'], shape=['Compactness2','Flatness'])
            extractor.enableFeaturesByName(
                firstorder=[
                    "Energy",
                    "TotalEnergy",
                    "Entropy",
                    "Minimum",
                    "Maximum",
                    "Mean",
                    "Kurtosis",
                ],
                shape=["Compactness2", "Flatness", "Maximum3DDiameter", "VoxelVolume"],
                glszm=["GrayLevelNonUniformity", "GrayLevelVariance"],
            )
            radiomicsresult = extractor.execute(image, seg)

            image, image_array = resampling_data(
                image,
                resampling_mode=resampling_mode,
                medianspacing=medianspacing,
                anisotropy_threshold=anisotropy_threshold,
                resize=resize,
                is_seg=False,
            )

            # get segmentation bounding box of primary tumor
            # check if image and segmentation have same dimensions
            seg, seg_array = resampling_data(
                seg,
                resampling_mode=resampling_mode,
                medianspacing=medianspacing,
                anisotropy_threshold=anisotropy_threshold,
                resize=resize,
                is_seg=True,
            )
            if image.GetSize() != seg.GetSize():
                print(
                    f"Warning: Image and segmentation have different dimensions for Subject ID = {subject_id} after resampling"
                )
            zmin_seg, ymin_seg, xmin_seg = np.where(seg_array > 0)
            zmax_seg, ymax_seg, xmax_seg = np.where(seg_array > 0)
            zmin_seg = np.min(zmin_seg)
            ymin_seg = np.min(ymin_seg)
            xmin_seg = np.min(xmin_seg)
            zmax_seg = np.max(zmax_seg)
            ymax_seg = np.max(ymax_seg)
            xmax_seg = np.max(xmax_seg)

            if len(lung_labels) > 1:
                # combine lung segmentations
                lungseg_array = np.zeros_like(image_array)
                for segment_number in lung_labels:
                    lungseg = result.segment_image(segment_number)
                    # resample lung segmentation to same spacing as image
                    lungseg, lungseg_array_part = resampling_data(
                        lungseg,
                        resampling_mode=resampling_mode,
                        medianspacing=medianspacing,
                        anisotropy_threshold=anisotropy_threshold,
                        resize=resize,
                        is_seg=True,
                    )
                    lungseg_array += (
                        lungseg_array_part  # dimension of lungseg_array is (z,y,x)
                    )

                # get lung segmentation bounding box
                zmin, ymin, xmin = np.where(lungseg_array > 0)
                zmax, ymax, xmax = np.where(lungseg_array > 0)
                zmin, ymin, xmin = np.min(zmin), np.min(ymin), np.min(xmin)
                zmax, ymax, xmax = np.max(zmax), np.max(ymax), np.max(xmax)

            else:
                print(
                    f"Warning: No lung segmentation found for Subject ID = {subject_id}"
                )
                zmin, ymin, xmin = -1, -1, -1
                zmax, ymax, xmax = -1, -1, -1

            # save metadata and radiomics features and clinical data
            data.append(
                {
                    "subject_id": subject_id,
                    "Compactness2": radiomicsresult["original_shape_Compactness2"],
                    "Flatness": radiomicsresult["original_shape_Flatness"],
                    "TotalEnergy": radiomicsresult["original_firstorder_TotalEnergy"],
                    "Entropy": radiomicsresult["original_firstorder_Entropy"],
                    "Energy": radiomicsresult["original_firstorder_Energy"],
                    "Maximum3DDiameter": radiomicsresult[
                        "original_shape_Maximum3DDiameter"
                    ],
                    "VoxelVolume": radiomicsresult["original_shape_VoxelVolume"],
                    "Kurtosis": radiomicsresult["original_firstorder_Kurtosis"],
                    "Maximum": radiomicsresult["original_firstorder_Maximum"],
                    "Mean": radiomicsresult["original_firstorder_Mean"],
                    "Minimum": radiomicsresult["original_firstorder_Minimum"],
                    "GrayLevelNonUniformity": radiomicsresult[
                        "original_glszm_GrayLevelNonUniformity"
                    ],
                    "GrayLevelVariance": radiomicsresult[
                        "original_glszm_GrayLevelVariance"
                    ],
                    "HLHGrayLevelNonUniformity": radiomicsresult[
                        "wavelet-HLH_glszm_GrayLevelNonUniformity"
                    ],
                    "HLHGrayLevelVariance": radiomicsresult[
                        "wavelet-HLH_glszm_GrayLevelVariance"
                    ],
                    "Survival.time": clinicaldata[
                        clinicaldata["PatientID"] == subject_id
                    ]["Survival.time"].values[0],
                    "deadstatus.event": clinicaldata[
                        clinicaldata["PatientID"] == subject_id
                    ]["deadstatus.event"].values[0],
                    "age": clinicaldata[clinicaldata["PatientID"] == subject_id][
                        "age"
                    ].values[0],
                    "gender": clinicaldata[clinicaldata["PatientID"] == subject_id][
                        "gender"
                    ].values[0],
                    "zmin": zmin,
                    "zmax": zmax,
                    "ymin": ymin,
                    "ymax": ymax,
                    "xmin": xmin,
                    "xmax": xmax,
                    "zmin_seg": zmin_seg,
                    "zmax_seg": zmax_seg,
                    "ymin_seg": ymin_seg,
                    "ymax_seg": ymax_seg,
                    "xmin_seg": xmin_seg,
                    "xmax_seg": xmax_seg,
                }
            )

            if get_tumour_bbox_patches:
                # crop image to tumour segmentation bounding box plus margin tumour_crop_margin
                zmin_seg_crop = max(zmin_seg - tumour_crop_margin[0], 0)
                zmax_seg_crop = min(
                    zmax_seg + tumour_crop_margin[0], image_array.shape[0]
                )
                ymin_seg_crop = max(ymin_seg - tumour_crop_margin[1], 0)
                ymax_seg_crop = min(
                    ymax_seg + tumour_crop_margin[1], image_array.shape[1]
                )
                xmin_seg_crop = max(xmin_seg - tumour_crop_margin[2], 0)
                xmax_seg_crop = min(
                    xmax_seg + tumour_crop_margin[2], image_array.shape[2]
                )
                image_array = image_array[
                    zmin_seg_crop:zmax_seg_crop,
                    ymin_seg_crop:ymax_seg_crop,
                    xmin_seg_crop:xmax_seg_crop,
                ]

                # save image as numpy array with shape (z,y,x)
                # create folder if not exists
                np.save(
                    os.path.join(
                        root, preprocessed_folder_name, f"{row['Subject ID']}.npy"
                    ),
                    image_array,
                )
                print(f"Image crop shape of {subject_id}: {image_array.shape}")

            else:
                # save image as numpy array with shape (z,y,x)
                # create folder if not exists
                np.save(
                    os.path.join(
                        root, preprocessed_folder_name, f"{row['Subject ID']}.npy"
                    ),
                    image_array,
                )
                print(
                    f"Image shape of {subject_id}: {sitk.GetArrayFromImage(image).shape}"
                )

    df_data = pd.DataFrame(data)
    # print shape of data
    print(
        f"Saving metadata of {df_data.shape} subjects to {os.path.join(root, preprocessed_folder_name, f'metadata.txt')}"
    )
    df_data.to_csv(
        os.path.join(root, preprocessed_folder_name, f"metadata.txt"),
        index=False,
        sep=" ",
    )


def preprocess_nsclc_radiomics_segmentations_only(
    folder="NSCLC-Radiomics",
    root="/absolute/path/to/datasets/NSCLC_Radiomics/",
    metadata_filename="metadata.csv",
    medianspacing=[0.9765625, 0.9765625, 3.0],
    resize=True,
    get_largest_tumour_component_only=False,
    get_tumour_bbox_patches=False,
    tumour_crop_margin=[2, 6, 6],
    anisotropy_threshold=3.0,
    resampling_mode="sitk_isotropic",
    preprocessed_folder_name=None,
    normalize_intensity=False,
    global_mean=-250.94877486254458,
    global_std=361.7110462718799,
    global_5p=-845.4839016685603,
    global_95p=158.39093935142375,
):
    if preprocessed_folder_name is None:
        if get_tumour_bbox_patches:
            preprocessed_folder_name = "preprocessed_tumourbbox_patches"
        else:
            preprocessed_folder_name = "preprocessed"

    if not os.path.exists(os.path.join(root, preprocessed_folder_name)):
        os.makedirs(os.path.join(root, preprocessed_folder_name))

    # save preprocessing parameters as json file
    with open(
        os.path.join(root, preprocessed_folder_name, f"preprocessing_parameters.json"),
        "w",
    ) as f:
        json.dump(
            {
                "medianspacing": medianspacing,
                "resize": resize,
                "get_largest_tumour_component_only": get_largest_tumour_component_only,
                "get_tumour_bbox_patches": get_tumour_bbox_patches,
                "tumour_crop_margin": tumour_crop_margin,
                "anisotropy_threshold": anisotropy_threshold,
                "resampling_mode": resampling_mode,
                "normalize_intensity": normalize_intensity,
                "global_mean": global_mean,
                "global_std": global_std,
                "global_5p": global_5p,
                "global_95p": global_95p,
            },
            f,
            indent=4,
        )

    metadata = pd.read_csv(
        os.path.join(root, metadata_filename),
    )

    for i, row in tqdm(
        metadata[metadata["Modality"] == "CT"].iterrows(),
        total=metadata[metadata["Modality"] == "CT"].shape[0],
    ):
        subject_id = row["Subject ID"]
        if subject_id not in [
            "LUNG1-014",
            "LUNG1-021",
            "LUNG1-085",
            "LUNG1-128",
        ]:  # filter out ids with missing slices
            filename = os.path.join(root, os.path.normpath(row["File Location"]))
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(filename)
            reader.SetFileNames(dicom_names)
            image = reader.Execute()  # dimensions are (x,y,z)

            segrow = metadata[
                (metadata["Modality"] == "SEG") & (metadata["Subject ID"] == subject_id)
            ]
            if segrow.shape[0] > 1:
                print(
                    f"Warning: More than one segmentation found for Subject ID = {subject_id}: {segrow.shape[0]}"
                )
            segrow = segrow.iloc[0]
            segfilename = os.path.join(
                root, os.path.normpath(segrow["File Location"]), "1-1.dcm"
            )

            dcm = pydicom.dcmread(segfilename)
            reader = pydicom_seg.SegmentReader()
            result = reader.read(dcm)

            tumor_labels = []
            lung_labels = []
            for segment_number in result.available_segments:
                if "GTV" in result.segment_infos[segment_number].SegmentDescription:
                    tumor_labels.append(segment_number)
                if "Lung" in result.segment_infos[segment_number].SegmentDescription:
                    lung_labels.append(segment_number)

            if len(tumor_labels) > 1:
                print(
                    f"Warning: More than one tumor segmentation found for Subject ID = {subject_id}: {len(tumor_labels)}"
                )

            segment_number = tumor_labels[0]
            seg = result.segment_image(segment_number)  # lazy construction
            # skip if image and segmentation have different dimensions
            if image.GetSize() != seg.GetSize():
                print(
                    f"Warning: Image and segmentation have different dimensions for Subject ID = {subject_id}. Skipping..."
                )
                continue

            if get_largest_tumour_component_only:
                # get largest tumour connected component only with sitk
                cc_seg = sitk.ConnectedComponent(seg)
                sorted_cc_seg = sitk.RelabelComponent(cc_seg, sortByObjectSize=True)
                seg = sitk.Cast(sorted_cc_seg == 1, sitk.sitkUInt8)

            seg, seg_array = resampling_data(
                seg,
                resampling_mode=resampling_mode,
                medianspacing=medianspacing,
                anisotropy_threshold=anisotropy_threshold,
                resize=resize,
                is_seg=True,
            )

            if image.GetSize() != seg.GetSize():
                print(
                    f"Warning: Image and segmentation have different dimensions for Subject ID = {subject_id} after resampling"
                )
            zmin_seg, ymin_seg, xmin_seg = np.where(seg_array > 0)
            zmax_seg, ymax_seg, xmax_seg = np.where(seg_array > 0)
            zmin_seg = np.min(zmin_seg)
            ymin_seg = np.min(ymin_seg)
            xmin_seg = np.min(xmin_seg)
            zmax_seg = np.max(zmax_seg)
            ymax_seg = np.max(ymax_seg)
            xmax_seg = np.max(xmax_seg)

            if get_tumour_bbox_patches:
                # crop image to tumour segmentation bounding box plus margin tumour_crop_margin
                zmin_seg_crop = max(zmin_seg - tumour_crop_margin[0], 0)
                zmax_seg_crop = min(
                    zmax_seg + tumour_crop_margin[0], seg_array.shape[0]
                )
                ymin_seg_crop = max(ymin_seg - tumour_crop_margin[1], 0)
                ymax_seg_crop = min(
                    ymax_seg + tumour_crop_margin[1], seg_array.shape[1]
                )
                xmin_seg_crop = max(xmin_seg - tumour_crop_margin[2], 0)
                xmax_seg_crop = min(
                    xmax_seg + tumour_crop_margin[2], seg_array.shape[2]
                )
                seg_array = seg_array[
                    zmin_seg_crop:zmax_seg_crop,
                    ymin_seg_crop:ymax_seg_crop,
                    xmin_seg_crop:xmax_seg_crop,
                ]

                # save image as numpy array with shape (z,y,x)
                # create folder if not exists
                np.save(
                    os.path.join(
                        root, preprocessed_folder_name, f"{row['Subject ID']}_SEG.npy"
                    ),
                    seg_array,
                )
                print(f"Segmentation crop shape of {subject_id}: {seg_array.shape}")

            else:
                # save image as numpy array with shape (z,y,x)
                # create folder if not exists
                np.save(
                    os.path.join(
                        root, preprocessed_folder_name, f"{row['Subject ID']}_SEG.npy"
                    ),
                    seg_array,
                )
                print(
                    f"Segmentation shape of {subject_id}: {sitk.GetArrayFromImage(seg).shape}"
                )


def split_nsclc_radiomics(
    train_val_test_split_size=[0.7, 0.1, 0.2],
    root="/absolute/path/to/datasets/NSCLC_Radiomics/",
    preprocessed_folder_name="preprocessed",
    metadata_filename="metadata.txt",
):
    suffix_list = ["CG", "EG", "train", "CGval", "EGval", "val", "test"]

    df_data = pd.read_csv(
        os.path.join(root, preprocessed_folder_name, metadata_filename), sep=" "
    )

    # split data into (train, val) and test
    df_trainval, df_test = train_test_split(
        df_data, test_size=train_val_test_split_size[2], random_state=1
    )
    # split data into train and val
    df_train, df_val = train_test_split(
        df_trainval,
        test_size=train_val_test_split_size[1]
        / (train_val_test_split_size[0] + train_val_test_split_size[1]),
        random_state=1,
    )

    # split train and val data into CG and EG
    df_train_CG, df_train_EG = train_test_split(df_train, test_size=0.5, random_state=1)
    df_val_CG, df_val_EG = train_test_split(df_val, test_size=0.5, random_state=1)

    # save data
    for df, suffix in zip(
        [df_train_CG, df_train_EG, df_train, df_val_CG, df_val_EG, df_val, df_test],
        suffix_list,
    ):
        print(
            f"Saving metadata of {df.shape} subjects to {os.path.join(root, preprocessed_folder_name, f'metadata_{suffix}.txt')}"
        )
        df.to_csv(
            os.path.join(root, preprocessed_folder_name, f"metadata_{suffix}.txt"),
            index=False,
            sep=" ",
        )


def combine_trainval_metadata(
    root="/absolute/path/to/datasets/NSCLC_Radiomics/",
    preprocessed_folder_name="preprocessed",
    suffix="",
    dtype=None,
):
    # combine CG and CGval metadata files into one
    df_CG = pd.read_csv(
        os.path.join(root, preprocessed_folder_name, f"metadata_CG{suffix}.txt"),
        sep=" ",
        dtype=dtype,
    )
    df_CGval = pd.read_csv(
        os.path.join(root, preprocessed_folder_name, f"metadata_CGval{suffix}.txt"),
        sep=" ",
        dtype=dtype,
    )

    df_CG_CGval = pd.concat([df_CG, df_CGval], ignore_index=True)
    df_CG_CGval.to_csv(
        os.path.join(root, preprocessed_folder_name, f"metadata_CGtraincv{suffix}.txt"),
        index=False,
        sep=" ",
    )

    # combine EG and EGval metadata files into one
    df_EG = pd.read_csv(
        os.path.join(root, preprocessed_folder_name, f"metadata_EG{suffix}.txt"),
        sep=" ",
        dtype=dtype,
    )
    df_EGval = pd.read_csv(
        os.path.join(root, preprocessed_folder_name, f"metadata_EGval{suffix}.txt"),
        sep=" ",
        dtype=dtype,
    )

    df_EG_EGval = pd.concat([df_EG, df_EGval], ignore_index=True)
    df_EG_EGval.to_csv(
        os.path.join(root, preprocessed_folder_name, f"metadata_EGtraincv{suffix}.txt"),
        index=False,
        sep=" ",
    )

    # check if metadata_train.txt and metadata_val.txt exist
    if not os.path.isfile(
        os.path.join(root, preprocessed_folder_name, f"metadata_train{suffix}.txt")
    ):
        # combine CG and EG into train metadata file
        df_train = pd.concat([df_CG, df_EG], ignore_index=True)
        df_train.to_csv(
            os.path.join(root, preprocessed_folder_name, f"metadata_train{suffix}.txt"),
            index=False,
            sep=" ",
        )

    else:
        # combine train and val metadata files into one
        df_train = pd.read_csv(
            os.path.join(root, preprocessed_folder_name, f"metadata_train{suffix}.txt"),
            sep=" ",
            dtype=dtype,
        )
    df_val = pd.read_csv(
        os.path.join(root, preprocessed_folder_name, f"metadata_val{suffix}.txt"),
        sep=" ",
        dtype=dtype,
    )

    df_trainval = pd.concat([df_train, df_val], ignore_index=True)
    df_trainval.to_csv(
        os.path.join(root, preprocessed_folder_name, f"metadata_traincv{suffix}.txt"),
        index=False,
        sep=" ",
    )


def combine_preprocessed_files(
    root="/absolute/path/to/datasets/NSCLC_Radiomics/",
    file_names=["CG", "CGval"],
    output_file="imgs_CGtraincv",
    preprocessed_folder_name="preprocessed",
    suffix="",
):
    data_combined = None

    for file_name in file_names:
        file_path = os.path.join(
            root, preprocessed_folder_name, f"{file_name}{suffix}.pt"
        )

        if os.path.isfile(file_path):
            data = torch.load(file_path)

            if data_combined is None:
                data_combined = data
            elif isinstance(data_combined, list):
                data_combined.extend(data)
            else:
                data_combined = torch.cat([data_combined, data], dim=0)

    if data_combined is not None:
        torch.save(
            data_combined,
            os.path.join(root, preprocessed_folder_name, f"{output_file}{suffix}.pt"),
        )

    print(f"Saved {output_file}{suffix}.pt")


# get metadata and normalize radiomics features
def get_metadata_and_normalize_radiomics_features(
    root="/absolute/path/to/datasets/NSCLC_Radiomics/",
    preprocessed_folder_name="preprocessed",
    metadata_filename="metadata.txt",
    normalization_method="zscore",
    radiomics_features_to_normalize=[
        "Compactness2",
        "Flatness",
        "TotalEnergy",
        "Entropy",
        "Energy",
        "Maximum3DDiameter",
        "VoxelVolume",
        "Kurtosis",
        "Maximum",
        "Mean",
        "Minimum",
        "GrayLevelNonUniformity",
        "GrayLevelVariance",
        "HLHGrayLevelNonUniformity",
        "HLHGrayLevelVariance",
    ],
):
    if metadata_filename is not None:
        df_data = pd.read_csv(
            os.path.join(root, preprocessed_folder_name, metadata_filename), sep=" "
        )
        metadata_df_list = [df_data]
        medata_filenames = [metadata_filename]

    else:
        # get all metadata files
        metadata_df_list = []
        medata_filenames = []

        for filename in os.listdir(os.path.join(root, preprocessed_folder_name)):
            if filename.startswith("metadata") and filename.endswith(".txt"):
                df_data = pd.read_csv(
                    os.path.join(root, preprocessed_folder_name, filename), sep=" "
                )
                metadata_df_list.append(df_data)
                medata_filenames.append(filename)

    # compute global mean and std of the radiomics features from the training set
    feature_means = {}
    feature_stds = {}
    for df_data, metadata_filename in zip(metadata_df_list, medata_filenames):
        # normalize radiomics features (z-scoring) and save as new columns in df_data
        if metadata_filename == "metadata_train.txt":
            # only save features not ending with _zscore or _zscoretr
            for feature in radiomics_features_to_normalize:
                feature_means[feature] = df_data[feature].mean()
                feature_stds[feature] = df_data[feature].std()

    for df_data, metadata_filename in zip(metadata_df_list, medata_filenames):
        # normalize radiomics features (z-scoring) and save as new columns in df_data
        for feature in radiomics_features_to_normalize:
            print(
                f"Normalizing {feature}: mean {df_data[feature].mean()} / global mean {feature_means[feature]}"
            )
            if normalization_method == "zscore":
                df_data[f"{feature}_zscore"] = (
                    df_data[feature] - df_data[feature].mean()
                ) / df_data[feature].std()
            elif normalization_method == "zscoretr":
                df_data[f"{feature}_zscoretr"] = (
                    df_data[feature] - feature_means[feature]
                ) / feature_stds[feature]
            elif normalization_method == "minmax":
                df_data[f"{feature}_minmax"] = (
                    df_data[feature] - df_data[feature].min()
                ) / (df_data[feature].max() - df_data[feature].min())
            else:
                raise NotImplementedError

        save_filename = os.path.join(
            root,
            preprocessed_folder_name,
            metadata_filename,
        )
        print(f"Saving metadata to {save_filename}")
        df_data.to_csv(save_filename, index=False, sep=" ")


if __name__ == "__main__":
    # CUB-200-2011
    root_path = os.path.join(
        os.getenv("DATASET_LOCATION", "/absolute/path/to/datasets"),
        "CUB_200_2011",
        "CUB_200_2011",
    )
    preprocess_img_data_cub2011(root=root_path, class_ids=None)
    preprocess_img_data_cub2011(
        root=root_path,
        class_ids=None,
        resize=False,
    )

    # CUB-200-2011: Combine CG and CGval files into one for 5-fold CV
    combine_preprocessed_files(
        root=root_path,
        preprocessed_folder_name="processed",
        suffix="_allclasses",
        file_names=["imgs_CG", "imgs_CGval"],
        output_file="imgs_CGtraincv",
    )

    combine_preprocessed_files(
        root_path,
        preprocessed_folder_name="processed",
        suffix="_allclasses",
        file_names=["imgs_EG", "imgs_EGval"],
        output_file="imgs_EGtraincv",
    )

    combine_preprocessed_files(
        root_path,
        preprocessed_folder_name="processed",
        suffix="_allclasses",
        file_names=["imgs_CG", "imgs_EG"],
        output_file="imgs_train",
    )

    combine_preprocessed_files(
        root_path,
        preprocessed_folder_name="processed",
        suffix="_allclasses",
        file_names=["imgs_train", "imgs_val"],
        output_file="imgs_traincv",
    )

    # ISIC 2018
    root_path = os.path.join(
        os.getenv("DATASET_LOCATION", "/absolute/path/to/datasets"), "ISIC2018"
    )
    preprocess_img_data_cub2011(root=root_path, class_ids=None)
    attributes_to_df_isic2018(
        "ISIC2018_Task2_Training_GroundTruth_v3",
        root=root_path,
    )
    attributes_to_df_isic2018(
        "ISIC2018_Task2_Validation_GroundTruth",
        root=root_path,
    )
    preprocess_img_data_isic2018(
        attr_filename="attributes_ISIC2018_Task2_Training_GroundTruth_v3.txt",
        root=root_path,
        get_trainval_split=True,
        preprocess_imgs=True,
    )
    preprocess_img_data_isic2018(
        attr_filename="attributes_ISIC2018_Task2_Validation_GroundTruth.txt",
        root=root_path,
        get_trainval_split=False,
        preprocess_imgs=True,
    )

    # ISIC 2018: Combine CG and CGval files into one for 5-fold CV
    for suffix in ["_resized", "_resized_normalized"]:
        combine_preprocessed_files(
            root=root_path,
            preprocessed_folder_name="preprocessed",
            suffix=suffix,
            file_names=["imgs_CG", "imgs_CGval"],
            output_file="imgs_CGtraincv",
        )

        combine_preprocessed_files(
            root=root_path,
            preprocessed_folder_name="preprocessed",
            suffix=suffix,
            file_names=["imgs_EG", "imgs_EGval"],
            output_file="imgs_EGtraincv",
        )

        combine_preprocessed_files(
            root=root_path,
            preprocessed_folder_name="preprocessed",
            suffix=suffix,
            file_names=["imgs_CG", "imgs_EG"],
            output_file="imgs_train",
        )

        combine_preprocessed_files(
            root=root_path,
            preprocessed_folder_name="preprocessed",
            suffix=suffix,
            file_names=["imgs_train", "imgs_val"],
            output_file="imgs_traincv",
        )

    # NSCLC-Radiomics
    root_path = os.path.join(
        os.getenv("DATASET_LOCATION", "/absolute/path/to/datasets"), "NSCLC_Radiomics"
    )

    # alternative to preprocessing based on nnunet
    preprocess_nsclc_radiomics(
        root=root_path,
        resize=False,
        get_tumour_bbox_patches=True,
        get_largest_tumour_component_only=True,
    )
    split_nsclc_radiomics(
        root=root_path,
        get_classification_labels=True,
        preprocessed_folder_name="preprocessed_tumourbbox_patches",
    )
    get_metadata_and_normalize_radiomics_features(
        root=root_path,
        preprocessed_folder_name="preprocessed_tumourbbox_patches",
        metadata_filename=None,
    )

    preprocess_nsclc_radiomics(
        root=root_path,
        resize=False,
        get_tumour_bbox_patches=True,
        get_largest_tumour_component_only=True,
        preprocessed_folder_name="preprocessed_tumourbbox_patches_nnunetresample",
        resampling_mode="nnunet",
    )
    split_nsclc_radiomics(
        root=root_path,
        get_classification_labels=True,
        preprocessed_folder_name="preprocessed_tumourbbox_patches_nnunetresample",
    )
    get_metadata_and_normalize_radiomics_features(
        root=root_path,
        preprocessed_folder_name="preprocessed_tumourbbox_patches_nnunetresample",
        normalization_method="zscore",
        metadata_filename=None,
    )

    # NSCLC-Radiomics: Combine CG and CGval files into one for 5-fold CV
    combine_trainval_metadata(
        root=root_path,
        preprocessed_folder_name="preprocessed_tumourbbox_patches_nnunetresample",
    )

    # Get preprocessed NSCLC-Radiomics Segmentations
    preprocess_nsclc_radiomics_segmentations_only(
        resize=False,
        get_tumour_bbox_patches=True,
        get_largest_tumour_component_only=True,
        preprocessed_folder_name="preprocessed_tumourbbox_patches_nnunetresample_seg",
        resampling_mode="nnunet",
    )

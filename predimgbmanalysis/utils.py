import matplotlib.pyplot as plt
import torch
import pydicom
import pydicom_seg
from pydicom import dcmread
from pydicom.data import get_testdata_file
import SimpleITK as sitk


def plot_figure(iterations, values, ylabel=None, labels=None):
    fig = plt.figure()

    # if isinstance(values, list) and len(values) > 1:
    for i, _ in enumerate(values):
        if labels is None:
            plt.plot(iterations, values[i], label=f"{i}")
        if isinstance(labels, list):
            plt.plot(iterations, values[i], label=labels[i])
    plt.legend()

    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.xlabel("training iterations")
    plt.close(fig)
    return fig


# %%
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_mean_and_std(dataloader, per_pixel_weighting=False):
    # "Each image has the same weight"
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    if per_pixel_weighting:
        for data, _ in dataloader:
            channels_sum += torch.sum(data, dim=[0, 2, 3])
            channels_squared_sum += torch.sum(data**2, dim=[0, 2, 3])
            num_batches += 1
            num_pixels += data.shape[2] * data.shape[3]

        mean = channels_sum / num_pixels
        std = (channels_squared_sum / num_pixels - mean**2) ** 0.5
    else:
        for data, _ in dataloader:
            # Mean over batch, height and width, but not over the channels
            channels_sum += torch.mean(data, dim=[0, 2, 3])
            channels_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
            num_batches += 1

        mean = channels_sum / num_batches
        std = (channels_squared_sum / num_batches - mean**2) ** 0.5

    return mean, std


def normalize_image(img):
    img = (img - img.min()) / (max(1.0, img.max()) - img.min())
    return img


def dicom2nifti(input_file, output_file):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(input_file)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, output_file)


def dicomseg2nifti(input_file, output_path):
    dcm = pydicom.dcmread(input_file)
    reader = pydicom_seg.SegmentReader()
    result = reader.read(dcm)

    for segment_number in result.available_segments:
        image_data = result.segment_data(segment_number)  # directly available
        print(image_data.shape)
        image = result.segment_image(segment_number)  # lazy construction
        sitk.WriteImage(image, f"{output_path}/seg-{segment_number}.nii.gz", True)


def update_nested_dict(original_dict, new_dict):
    for key, value in new_dict.items():
        if (
            isinstance(value, dict)
            and key in original_dict
            and isinstance(original_dict[key], dict)
        ):
            # If both the original and new values are dictionaries, recursively update them
            original_dict[key] = update_nested_dict(original_dict[key], value)
        else:
            # If the key exists in the original dictionary, update the value; otherwise, add a new key-value pair
            original_dict[key] = value
    return original_dict

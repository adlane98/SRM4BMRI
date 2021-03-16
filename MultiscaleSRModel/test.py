import os
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from tensorflow.keras.models import load_model
# import SimpleITK as sitk

from model import psnr_model
from preprocessing import downsample
from utils.adamLRM import AdamLRM
from utils.utils import get_path


def write_output(input_path, output, output_path, affine=None):
    if output_path is None:
        output_path = fr"{get_path('outputs')}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    file_stem = Path(input_path).stem
    file_stem = file_stem[2:] if file_stem.startswith(r"\\") else file_stem

    output_name = fr"{file_stem}-output.nii"

    # sitk_image = sitk.GetImageFromArray(output[0, :, :, :, 0])
    # sitk.WriteImage(sitk_image, str(Path(output_path) / output_name))

    new_image = nib.Nifti1Image(
        output[0, :, :, :, 0].astype(np.float64), affine=affine
    )
    nib.save(new_image, str(Path(output_path) / output_name))


def load_input(input_path):
    # input_nifti = sitk.ReadImage(input_path)
    # input_image = sitk.GetArrayFromImage(input_nifti)
    input_nifti = nib.load(input_path)
    input_image = input_nifti.get_data()

    return input_image[np.newaxis, :, :, :, np.newaxis], input_nifti.affine


def load_input_preproc(
        input_path, blur_sigma, downsampling_scale, interpolation_order
):
    # input_nifti = sitk.ReadImage(input_path)
    # input_image = sitk.GetArrayFromImage(input_nifti)

    input_nifti = nib.load(input_path)
    input_image = input_nifti.get_data()

    image, _ = downsample(
        input_image,
        downsampling_scale,
        interpolation_order,
        blur_sigma
    )
    return image[np.newaxis, :, :, :, np.newaxis], input_nifti.affine


def launch_testing(
        path_model,
        input_folder,
        output_folder=None,
        zoom_scale=None,
        downsample_scale=None,
):
    if zoom_scale is None and downsample_scale is None:
        zoom_scale = (2, 2, 2)

    model = load_model(
        path_model,
        custom_objects={"psnr_model": psnr_model, "AdamLRM": AdamLRM}
    )
    input_pathes = list(Path(input_folder).glob("*.nii*"))
    input_pathes = [str(p) for p in input_pathes]
    for input_path in input_pathes:
        test_input, aff = load_input(input_path)
        if zoom_scale is not None:
            zoomed_image = zoom(
                test_input[0, :, :, :, 0],
                zoom=zoom_scale,
                order=3
            )
            zoomed_image = zoomed_image[np.newaxis, :, :, :, np.newaxis]
            model_input = zoomed_image
        else:
            downsampled_image, _ = downsample(input_path, downsample_scale)
            model_input = downsampled_image[np.newaxis, :, :, :, np.newaxis]

        output = model.predict(model_input)

        write_output(input_path, output, output_folder, affine=aff)


if __name__ == '__main__':
    launch_testing(
        r"D:\OneDrive\Bureau\20210106-205104_model",
        r"D:\Projet\SRM4BMRI\MultiscaleSRModel\data2",
        r"D:\Projet\SRM4BMRI\MultiscaleSRModel\outputs",
    )

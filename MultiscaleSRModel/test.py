import os
from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model
import SimpleITK as sitk

from preprocessing import downsample
from utils.utils import get_path


def write_output(input_path, output, output_path):
    if output_path is None:
        output_path = fr"{get_path('outputs')}"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    output_name = fr"{Path(input_path).stem}-output.nii"
    sitk_image = sitk.GetImageFromArray(output[0, :, :, :, 0])
    sitk.WriteImage(sitk_image, fr"{output_path}\\{output_name}")


def load_input(input_path):
    input_nifti = sitk.ReadImage(input_path)
    input_image = sitk.GetArrayFromImage(input_nifti)

    return input_image[np.newaxis, :, :, :, np.newaxis]


def load_input_preproc(
        input_path, blur_sigma, downsampling_scale, interpolation_order
):
    input_nifti = sitk.ReadImage(input_path)
    image = sitk.GetArrayFromImage(input_nifti)

    image, _ = downsample(
        image,
        blur_sigma,
        downsampling_scale,
        interpolation_order
    )
    return image[np.newaxis, :, :, :, np.newaxis]


def launch_testing(
        path_model,
        input_folder,
        output_folder=None,
        preproc=False,
        blur_sigma=1,
        scales=(2, 2, 2),
        interpolation_order=3
):
    model = load_model(path_model)
    input_pathes = list(Path(input_folder).glob("*.nii*"))
    input_pathes = [str(p) for p in input_pathes]
    for input_path in input_pathes:
        if preproc:
            test_input = load_input_preproc(
                input_path, blur_sigma, scales, interpolation_order
            )
        else:
            test_input = load_input(input_path)

        output = model.predict(test_input)

        write_output(input_path, output, output_folder)


if __name__ == '__main__':
    launch_testing(
        r"D:\OneDrive\Bureau\20210106-205104_model",
        r"D:\Projet\SRM4BMRI\MultiscaleSRModel\data2",
        r"D:\Projet\SRM4BMRI\MultiscaleSRModel\outputs",
        preproc=True
    )

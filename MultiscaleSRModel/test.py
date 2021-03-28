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


def write_output(output, output_path, output_name, affine=None):
    new_image = nib.Nifti1Image(
        output.astype(np.float64), affine=affine
    )
    nib.save(new_image, str(Path(output_path) / f"{output_name}.nii"))


def load_input(input_path):
    # input_nifti = sitk.ReadImage(input_path)
    # input_image = sitk.GetArrayFromImage(input_nifti)
    input_nifti = nib.load(input_path)
    input_image = input_nifti.get_data()

    return input_image[np.newaxis, :, :, :, np.newaxis], input_nifti.affine


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

        image_name = Path(input_path).stem.split('.')[0]
        image_name = image_name[2:] if image_name.startswith(r"\\") else image_name
        image_folder = Path(output_folder) / image_name
        os.makedirs(image_folder, exist_ok=True)

        test_input, aff = load_input(input_path)
        if zoom_scale is not None:
            zoomed_image = zoom(
                test_input[0, :, :, :, 0],
                zoom=zoom_scale,
                order=3
            )
            write_output(zoomed_image, image_folder, "interpolated", aff)

            zoomed_image = zoomed_image[np.newaxis, :, :, :, np.newaxis]
            model_input = zoomed_image
        else:
            ref_image, downsampled_image, interpolated_image = downsample(
                input_path, downsample_scale
            )
            write_output(ref_image, image_folder, "tested", aff)
            write_output(downsampled_image, image_folder, "downsampled", aff)
            write_output(interpolated_image, image_folder, "interpolated", aff)

            model_input = interpolated_image[np.newaxis, :, :, :, np.newaxis]

        output = model.predict(model_input)
        write_output(output[0, :, :, :, 0], image_folder, "output", aff)


if __name__ == '__main__':
    launch_testing(
        r"D:\OneDrive\Bureau\20210106-205104_model",
        r"D:\Projet\SRM4BMRI\MultiscaleSRModel\data2",
        r"D:\Projet\SRM4BMRI\MultiscaleSRModel\outputs",
    )

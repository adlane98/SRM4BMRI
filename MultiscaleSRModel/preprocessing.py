import os

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from utils.patches import array_to_patches
from utils.store2hdf5 import store2hdf53D
from utils import write_metadata, get_hdf5_path, get_time, get_metadata_path
from utils.utils3D import imadjust3D, modcrop3D


def downsample(
        img_path,
        blur_sigma=1,
        downsampling_scale=(2, 2, 2),
        interpolation_order=3,
):

    reference_nifti = sitk.ReadImage(img_path)
    reference_image = sitk.GetArrayFromImage(reference_nifti)
    reference_image = np.swapaxes(reference_image, 0, 2).astype('float32')

    # Normalisation and modcrop
    reference_image = imadjust3D(reference_image, [0, 1])
    reference_image = modcrop3D(reference_image, downsampling_scale)

    # Blur and downsampling
    blur_reference_image = gaussian_filter(reference_image, sigma=blur_sigma)
    low_resolution_image = zoom(
        blur_reference_image,
        zoom=(1 / float(idxScale) for idxScale in downsampling_scale),
        order=interpolation_order
    )

    # Interpolation
    interpolated_image = zoom(
        low_resolution_image,
        zoom=downsampling_scale,
        order=interpolation_order
    )

    return interpolated_image, reference_image


def make_patches(
        data_image,
        label_image,
        patch_size=21,
        patch_stride=10,
        max_number_patches=3200,
):
    data_patches = array_to_patches(
        data_image,
        patch_shape=(patch_size, patch_size, patch_size),
        extraction_step=patch_stride,
        normalization=False
    )

    labels_patches = array_to_patches(
        label_image,
        patch_shape=(patch_size, patch_size, patch_size),
        extraction_step=patch_stride,
        normalization=False
    )

    # Add channel axis
    data_patches = data_patches[:, :, :, :, np.newaxis]
    labels_patches = labels_patches[:, :, :, :, np.newaxis]

    np.random.seed(0)  # makes the random numbers predictable
    random_order = np.random.permutation(data_patches.shape[0])

    data_patches = data_patches[random_order, :, :, :, :]
    labels_patches = labels_patches[random_order, :, :, :, :]

    return data_patches[:max_number_patches, :, :, :, :], labels_patches[:max_number_patches, :, :, :, :]


def prepare_data(
        images_path,

        # Downsample arguments
        blur_sigma=1,
        scales=None,
        interpolation_order=3,

        # Patches argument
        patch_size=21,
        patch_stride=10,
        max_number_patches_per_subject=3200,
):
    if len(images_path) == 0:
        raise Exception("No image to prepare.")

    if scales is None:
        scales = [(2, 2, 2)]

    time = get_time()

    list_hdf5_file_name = fr"{get_hdf5_path()}{time}.txt"
    os.makedirs(os.path.dirname(list_hdf5_file_name), exist_ok=True)
    with open(list_hdf5_file_name, "w") as lfh:
        for i, image_path in enumerate(images_path):
            hdf5_file_name = fr"{get_hdf5_path()}{time}_{Path(image_path).stem}.h5"
            lfh.write(fr"{hdf5_file_name}\n")

            data, label = [], []
            for j, scale in enumerate(scales):
                data_ds, label_ds = downsample(
                    image_path, blur_sigma, scale, interpolation_order
                )
                data_patches, label_patches = make_patches(
                    data_ds,
                    label_ds,
                    patch_size,
                    patch_stride,
                    max_number_patches_per_subject
                )

                data.append(data_patches)
                label.append(label_patches)

            data = np.concatenate(data, axis=0)
            label = np.concatenate(label, axis=0)
            store2hdf53D(hdf5_file_name, data, label, create=True)

    json_file_name = fr"{get_metadata_path()}{time}_preproc_parameter.json"
    write_metadata(
        json_file_name,
        {
           "images": images_path,
           "blur": blur_sigma,
           "scales": scales,
           "interpolation_order": interpolation_order,
           "patch_size": patch_size,
           "patch_stride": patch_stride,
           "max_number_patches_per_subject": max_number_patches_per_subject
        },
    )

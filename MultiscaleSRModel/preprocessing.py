import SimpleITK as sitk
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from patches import array_to_patches
from utils3D import imadjust3D, shave3D, modcrop3D


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

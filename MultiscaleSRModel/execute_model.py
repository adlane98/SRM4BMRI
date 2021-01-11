import argparse
from ast import literal_eval as make_tuple
import json
import os

import numpy as np
from pathlib import Path

from model import launch_training_hdf5
from preprocessing import downsample, make_patches
from store2hdf5 import store2hdf53D
from utils import get_time, write_metadata, get_metadata_path, get_hdf5_path


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
        scales = [(2, 2, 2), (3, 3, 3)]

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


def parsing():
    parser = argparse.ArgumentParser("Prepare data or launch training for "
                                     "MultiscaleSR model")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", help="Prepare data", action="store_true")
    group.add_argument("--launch", help="Launch training", action="store_true")

    # Downsampling args
    parser.add_argument("--mri", type=str, help="Path for one MRI.",
                        action="append")
    parser.add_argument("--sigma",
                        help="Standard deviation (sigma) of "
                             "Gaussian blur (default=1)",
                        type=int, default=1)
    parser.add_argument("-s", "--scale",
                        help="Scale factor (default = 2,2,2). Append mode: "
                             "-s 2,2,2 -s 3,3,3",
                        type=str, action="append")
    parser.add_argument("--order",
                        help="Order of spline interpolation (default=3) ",
                        type=int, default=3)
    parser.add_argument("-p", "--patchsize",
                        help="Indicates input patch size for extraction",
                        type=int, default=21)
    parser.add_argument("--stride",
                        help="Indicates step size at which extraction shall "
                             "be performed (default=10)",
                        type=int, default=10)
    parser.add_argument("--samples",
                        help="Indicates limit of samples in HDF5 file ",
                        type=int, default=3200)

    # Training
    parser.add_argument("-i", "--input",
                        help="Indicates file where is stored the list of hdf5 "
                             "files to use as input for training "
                             "(default=the last one in hdf5 folder)",
                        type=str, default=None)
    parser.add_argument("-l", "--layers",
                        help="Indicates number of layers of network "
                             "(default=10)",
                        type=int, default=10)
    parser.add_argument("--numkernel",
                        help="Indicates number of filters (default=64)",
                        type=int, default=64)
    parser.add_argument("-k", "--kernel",
                        help="Indicates size of filter (default=3)", type=int,
                        default=3)
    parser.add_argument("--kernelpad",
                        help="Indicates kernel padding (default=1)", type=int,
                        default=1)
    parser.add_argument("--epochs",
                        help="Indicates epochs of the training (default=20)",
                        type=int,
                        default=20)
    parser.add_argument("-b", "--batch",
                        help="Indicates batch size for HDF5 storage",
                        type=int, default=64)
    parser.add_argument("--adam",
                        help="Indicates Adam learning rate (default=0.0001)",
                        type=float, default=0.0001)

    args = parser.parse_args()

    if args.prepare:
        if args.scale is None:
            args.scale = [(2, 2, 2)]
        else:
            for idx in range(0, len(args.scale)):
                args.scale[idx] = make_tuple(args.scale[idx])
                if np.isscalar(args.scale[idx]):
                    args.scale[idx] = (
                        args.scale[idx], args.scale[idx], args.scale[idx])
                else:
                    if len(args.scale[idx]) != 3:
                        raise AssertionError("Not support this scale factor !")

        if args.mri is None:
            raise AssertionError("No image to prepare.")

    return args


if __name__ == '__main__':
    args = parsing()

    if args.prepare:
        prepare_data(
            args.mri,
            blur_sigma=args.sigma,
            scales=args.scale,
            interpolation_order=args.order,
            patch_size=args.patchsize,
            patch_stride=args.stride,
            max_number_patches_per_subject=args.samples
        )
    else:
        launch_training_hdf5(
            args.input,
            depth=args.layers,
            nb_filters=args.numkernel,
            kernel_size=args.kernel,
            padding=args.kernelpad,
            epochs=args.epochs,
            batch_size=args.batch,
            adam_lr=args.adam
        )

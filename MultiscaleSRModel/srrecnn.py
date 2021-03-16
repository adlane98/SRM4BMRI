import argparse
from ast import literal_eval as make_tuple

import numpy as np

from model import launch_training_hdf5
from preprocessing import prepare_data
from test import launch_testing


def parsing():
    parser = argparse.ArgumentParser("Prepare data or launch training for "
                                     "MultiscaleSR model")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare", help="Prepare data", action="store_true")
    group.add_argument("--train", help="Launch training", action="store_true")
    group.add_argument("--test", help="Launch testing", action="store_true")

    # Downsampling args
    parser.add_argument("--mri", help="Folder where are stored MRI files",
                        type=str)
    parser.add_argument("--sigma",
                        help="Standard deviation (sigma) of "
                             "Gaussian blur (default=1)",
                        type=float, default=1.0)
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

    # Testing
    parser.add_argument("--model",
                        help="Path of the trained model to test",
                        type=str, default=None)
    parser.add_argument("--testinput",
                        help="Folder of images to test",
                        type=str, default=None)
    parser.add_argument("--output",
                        help="Folder where to store outputs Nifti files",
                        type=str, default=None)

    # We choose either to upsample the tested images (with zoom argument)
    # or to downsample and then upsample it (with downsample argument) to get
    # an input image and an output image with the same size.
    # If the zoom argument is passed then the downsample argument is ignored.
    # If neither is set, then, by default it will be a [2,2,2] zoom scale.
    parser.add_argument("-z", "--zoom",
                        help="Zoom scale of the test image.",
                        default=None)
    parser.add_argument("-d", "--downsample",
                        help="Downsampling scale",
                        default=None)

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

    if args.test:
        if args.zoom:
            args.zoom = make_tuple(args.zoom)
        if args.downsample:
            args.downsample = make_tuple(args.downsample)

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
    elif args.train:
        launch_training_hdf5(
            args.input,
            depth=args.layers,
            nb_filters=args.numkernel,
            kernel_size=args.kernel,
            epochs=args.epochs,
            batch_size=args.batch,
            adam_lr=args.adam
        )
    else:
        launch_testing(
            path_model=args.model,
            input_folder=args.testinput,
            output_folder=args.output,
            zoom_scale=args.zoom,
            downsample_scale=args.downsample
        )

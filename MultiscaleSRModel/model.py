from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import image, pad
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.layers import Add, Conv3D, Input, ReLU
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils.adamLRM import AdamLRM
from utils.utils import get_path, get_time, read_hdf5_files, write_metadata


def psnr_model(y_pred, y_true):
    return image.psnr(y_pred.numpy(), y_true, np.max(y_pred.numpy())).numpy()


def SRReCNN3D(input_shape, depth, nb_filters, kernel_size, padding):
    input_layer = Input(input_shape)
    layer = input_layer

    for i in range(depth+1):
        nf = 1 if i == depth else nb_filters
        padded_layer = pad(layer, [[0, 0], [padding, padding], [padding, padding], [padding, padding], [0, 0]])
        layer = Conv3D(
            filters=nf,
            kernel_size=kernel_size,
            strides=1,
            padding="valid",
            kernel_initializer=RandomNormal(
                mean=0,
                stddev=np.sqrt(2.0/float(nb_filters * kernel_size ** 3))
            ),
            bias_initializer=Constant(0)
        )(padded_layer)
        if i < depth:
            layer = ReLU()(layer)

    final_layer = Add()([input_layer, layer])
    #final_layer = Flatten()(final_layer)

    model = Model(input_layer, final_layer)

    return model


def launch_training(
        data,
        labels,
        depth=10,
        nb_filters=64,
        kernel_size=3,
        epochs=20,
        batch_size=4,
        adam_lr=0.0001,
        hdf5_source_file=None
):
    launching_time = get_time()

    if kernel_size % 2 != 1 or kernel_size <= 1 or kernel_size >= data.shape[1]:
        raise AttributeError(
            "The kernel size (-k, --kernel) needs to be odd, "
            "greater than 1 and smaller than the patch size."
        )

    padding = (kernel_size - 1)/2
    padding = int(padding)

    json_file_name = fr"{get_path('metadata')}{launching_time}_training_parameter.json"
    write_metadata(
        json_file_name,
        {
            "hdf5_source_file": hdf5_source_file,
            "depth": depth,
            "nb_filters": nb_filters,
            "kernel_size": kernel_size,
            "padding": padding,
            "epochs": epochs,
            "batch_size": batch_size,
            "adam_lr": adam_lr,
            "nb_samples": data.shape[0],
            "sample_dimension": data.shape[1:]
        },
    )

    model = SRReCNN3D(data[0].shape, depth, nb_filters, kernel_size, padding)
    model.compile(
        optimizer=Adam(),
        loss="mse",
        metrics=[psnr_model],
        run_eagerly=True
    )
    history = model.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs
    )

    weights_file_name = fr"{get_path('metadata')}{launching_time}_weights.h5"
    model_file_name = fr"{get_path('metadata')}{launching_time}_model"
    model.save_weights(weights_file_name, save_format="hdf5")
    model.save(model_file_name)
    draw_loss_and_psnr(history, launching_time)

    return model, history


def get_source_file(filename):
    if filename is not None:
        return filename
    else:
        hdf5_files = Path(get_path('hdf5')).glob("*.txt")
        hdf5_files = sorted(hdf5_files)
        return fr"{get_path('hdf5')}{hdf5_files[-1].name}"


def launch_training_hdf5(
        hdf5_source_file,
        depth=10,
        nb_filters=64,
        kernel_size=3,
        epochs=20,
        batch_size=4,
        adam_lr=0.0001
):
    hdf5_source_file = get_source_file(hdf5_source_file)
    data, labels = read_hdf5_files(hdf5_source_file)

    return launch_training(
        data,
        labels,
        depth,
        nb_filters,
        kernel_size,
        epochs,
        batch_size,
        adam_lr,
        hdf5_source_file,
    )


def draw_loss_and_psnr(history, launching_time):
    plt.figure(figsize=(11, 3))

    # Plot loss function
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['loss'])
    plt.title('loss')

    # Plot PSNR metric
    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, history.history['psnr_model'])
    plt.title('psnr')

    plt.savefig(fr"{get_path('metadata')}loss-psnr-curves-{launching_time}")

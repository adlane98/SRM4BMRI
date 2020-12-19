from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import image, pad
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.layers import Add, Conv3D, Input, ReLU
from tensorflow.keras.models import Model

from adamLRM import AdamLRM
from utils import get_time, read_hdf5_files


def psnr_model(y_pred, y_true):
    return image.psnr(y_pred.numpy(), y_true, np.max(y_pred.numpy())).numpy()


def SRReCNN3D(input_shape, depth, nb_filters, kernel_size, padding, to_json=False):
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

    if to_json:
        with open(f"model-{get_time()}.js", "w") as json_model:
            json_model.write(model.to_json())

    return model


def launch_training(
        data,
        labels,
        depth=10,
        nb_filters=64,
        kernel_size=3,
        padding=1,
        epochs=20,
        batch_size=4,
        adam_lr=0.0001
):
    model = SRReCNN3D(data[0].shape, depth, nb_filters, kernel_size, padding)
    model.compile(
        optimizer=AdamLRM(learning_rate=adam_lr),
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

    return model, history


def get_source_file(filename):
    if filename is not None:
        return filename
    else:
        hdf5_files = Path("hdf5\\").glob("*.txt")
        hdf5_files = sorted(hdf5_files)
        return f"hdf5\\{hdf5_files[-1].name}"


def launch_training_hdf5(
        hdf5_source_file,
        depth=10,
        nb_filters=64,
        kernel_size=3,
        padding=1,
        epochs=20,
        batch_size=4,
        adam_lr=0.0001
):
    hdf5_source_file = get_source_file(hdf5_source_file)
    data, labels = read_hdf5_files(hdf5_source_file)
    return launch_training(data, labels, depth, nb_filters, kernel_size, padding, epochs, batch_size, adam_lr)


def draw_loss_and_psnr(history):
    plt.figure(figsize=(11, 3))

    # Plot loss function
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['loss'])
    plt.title('loss')

    # Plot PSNR metric
    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, history.history['psnr_model'])
    plt.title('psnr')

    plt.savefig(f"loss-psnr-curve-{get_time()}")

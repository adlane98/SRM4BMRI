import numpy as np
import scipy.ndimage
import tensorflow as tf
from tensorflow import keras
from tensorflow import pad
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.layers import Conv3D, ReLU, Input, Add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import h5py
import SimpleITK as sitk

from adamLRM import AdamLRM
from patches import array_to_patches
from store2hdf5 import store2hdf53D
from utils3D import modcrop3D, shave3D, imadjust3D
import matplotlib.pyplot as plt


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
                stddev=0.0001
            ),
            bias_initializer=Constant(0)
        )(padded_layer)
        if i < depth:
            layer = ReLU()(layer)

    add_layer = Add()([input_layer, layer])

    #flat_layer = Flatten()(add_layer)
    #model = Model(input_layer, flat_layer)

    model = Model(input_layer, add_layer)

    if to_json:
        with open("model.js", "w") as json_model:
            json_model.write(model.to_json())

    return model


def launch_training(
        hdf5file, depth, nb_filters, kernel_size, padding
):
    with h5py.File(hdf5file) as hdf5data:
        data = list(hdf5data[list(hdf5data.keys())[0]])
        labels = list(hdf5data[list(hdf5data.keys())[1]])


    model = SRReCNN3D(data[0].shape, depth, nb_filters, kernel_size, padding)
    model.compile(optimizer=AdamLRM(learning_rate=0.0001), loss="mse")
    history = model.fit(np.asarray(data), np.asarray(labels).reshape(len(labels), ps**3), batch_size=4, epochs=20)
    plt.figure(figsize=(11, 3))

    # affichage de la valeur de la fonction de perte
    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, history.history['loss'])
    plt.title('loss')

    # affichage de la précision de notre réseau sur les données d'apprentissage
    plt.subplot(1, 2, 2)
    plt.plot(history.epoch, history.history['accuracy'])
    plt.title('accuracy')

    return model


def psnr_model(y_pred, y_true):
    return tf.image.psnr(y_pred.numpy(), y_true, np.max(y_pred.numpy())).numpy()

def psnr(y_pred, y_true):
    return tf.image.psnr(y_pred, y_true, np.max(y_pred))

# Compile the model
if __name__ == '__main__1':
    ps = 21
    model = SRReCNN3D([ps, ps, ps, 1], depth=10, nb_filters=64, kernel_size=3, padding=1)
    model.compile(optimizer=AdamLRM(learning_rate=0.0001), loss="mse", metrics=[psnr], run_eagerly=True)

    x1 = np.random.random((ps, ps, ps, 1))
    x2 = np.random.random((ps, ps, ps, 1))
    # y1 = np.random.random((ps**3))
    # y2 = np.random.random((ps**3))
    y1 = np.random.random((ps, ps, ps, 1))
    y2 = np.random.random((ps, ps, ps, 1))
    # print(psnr(x1[:, :, :, 0], y1[:, :, :, 0]))
    history = model.fit(x=np.array([x1, x2]), y=np.array([y1, y2]), batch_size=1)

    g = model.get_weights()

# HDF5
if __name__ == '__main__2':
    ps = 21
    bs = 64
    cs = 1

    x1 = np.random.random((bs, ps, ps, ps, cs))
    y1 = np.random.random((bs, ps, ps, ps, cs))

    store2hdf53D("data.h5", x1, y1)

    with h5py.File("data.h5") as hdf5data:
        data = list(hdf5data[list(hdf5data.keys())[0]])

    # bs = 8
    # data_batch = [data[i:i + bs] for i in range(0, len(data), bs)]
    # print(data_batch)


# Launch training
if __name__ == '__main__':
    sg = 1
    scale = (2, 2, 2)
    border = (3, 3, 10)
    order = 3
    ps = 21
    stride = 10
    hdf5_data = []
    hdf5_labels = []

    reference_nifti = sitk.ReadImage("1010.nii")

    # Get data from NIFTI
    reference_image = np.swapaxes(
        sitk.GetArrayFromImage(reference_nifti), 0, 2
    ).astype('float32')

    # Normalization
    reference_image = imadjust3D(reference_image, [0, 1])

    # ===== Generate input LR image =====
    # Blurring
    blur_reference_image = scipy.ndimage.filters.gaussian_filter(reference_image,
                                                                 sigma=sg)


    # Modcrop to scale factor
    blur_reference_image = modcrop3D(blur_reference_image, scale)
    reference_image = modcrop3D(reference_image, scale)

    # Downsampling
    low_resolution_image = scipy.ndimage.zoom(blur_reference_image,
                                              zoom=(1 / float(idxScale) for idxScale in scale),
                                              order=order)

    # Cubic Interpolation
    interpolated_image = scipy.ndimage.zoom(low_resolution_image,
                                            zoom=scale,
                                            order=order)

    # Shave border
    label_image = shave3D(reference_image, border)
    data_image = shave3D(interpolated_image, border)

    # Extract 3D patches
    data_patch = array_to_patches(data_image,
                                  patch_shape=(ps, ps, ps),
                                  extraction_step=stride,
                                  normalization=False)

    label_patch = array_to_patches(label_image,
                                   patch_shape=(ps, ps, ps),
                                   extraction_step=stride,
                                   normalization=False)

    hdf5_data.append(data_patch)
    hdf5_labels.append(label_patch)

    hdf5_data = np.asarray(hdf5_data)
    hdf5_data = hdf5_data.reshape((-1, ps, ps, ps))
    hdf5_labels = np.asarray(hdf5_labels).reshape((-1, ps, ps, ps))

    # Add channel axis !
    hdf5_data = hdf5_data[:, :, :, :, np.newaxis]
    hdf5_labels = hdf5_labels[:, :, :, :, np.newaxis]

    # Rearrange
    np.random.seed(0)  # makes the random numbers predictable
    random_order = np.random.permutation(hdf5_data.shape[0])
    hdf5_data = hdf5_data[random_order, :, :, :, :]
    hdf5_labels = hdf5_labels[random_order, :, :, :, :]

    # samples = 162
    # HDF5Datas = HDF5Datas[:samples, :, :, :, :]
    # HDF5Labels = HDF5Labels[:samples, :, :, :, :]

    hdf5name = "1010.h5"
    start_location = {'dat': (0, 0, 0, 0, 0), 'lab': (0, 0, 0, 0, 0)}
    current_data_location = store2hdf53D(filename=hdf5name,
                                         datas=hdf5_data,
                                         labels=hdf5_labels,
                                         startloc=start_location,
                                         chunksz=64)

    launch_training(hdf5name, depth=10, nb_filters=64, kernel_size=3, padding=1)


# Test human data
if __name__ == '__main__4':
    reference_nifti = sitk.ReadImage("KKI2009-01-MPRAGE.nii")

    # Get data from NIFTI
    print(sitk.GetArrayFromImage(reference_nifti).shape)


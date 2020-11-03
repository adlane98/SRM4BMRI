import numpy as np
import scipy.ndimage
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

    flat_layer = Flatten()(add_layer)

    model = Model(input_layer, flat_layer)

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
    plt.title('accuracy');


if __name__ == '__main__1':
    ps = 21
    model = SRReCNN3D([ps, ps, ps, 1], depth=10, nb_filters=64, kernel_size=3, padding=1)
    model.compile(optimizer=AdamLRM(learning_rate=0.0001), loss="mse")

    x1 = np.random.random((ps, ps, ps, 1))
    x2 = np.random.random((ps, ps, ps, 1))
    y1 = np.random.random((ps**3))
    y2 = np.random.random((ps**3))

    model.fit(x=np.array([x1, x2]), y=np.array([y1, y2]), batch_size=1)

    g = model.get_weights()

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


if __name__ == '__main__':
    sg = 1
    scale = (2, 2, 2)
    border = (3, 3, 10)
    order = 3
    ps = 21
    stride = 10
    HDF5Datas = []
    HDF5Labels = []

    ReferenceNifti = sitk.ReadImage("1010.nii")

    # Get data from NIFTI
    ReferenceImage = np.swapaxes(
        sitk.GetArrayFromImage(ReferenceNifti), 0, 2
    ).astype('float32')

    # Normalization
    ReferenceImage = imadjust3D(ReferenceImage, [0, 1])

    # ===== Generate input LR image =====
    # Blurring
    BlurReferenceImage = scipy.ndimage.filters.gaussian_filter(ReferenceImage,
                                                               sigma=sg)


    # Modcrop to scale factor
    BlurReferenceImage = modcrop3D(BlurReferenceImage, scale)
    ReferenceImage = modcrop3D(ReferenceImage, scale)

    # Downsampling
    LowResolutionImage = scipy.ndimage.zoom(BlurReferenceImage,
                                            zoom=(1 / float(idxScale) for idxScale in scale),
                                            order=order)

    # Cubic Interpolation
    InterpolatedImage = scipy.ndimage.zoom(LowResolutionImage,
                                           zoom=scale,
                                           order=order)

    # Shave border
    LabelImage = shave3D(ReferenceImage, border)
    DataImage = shave3D(InterpolatedImage, border)

    # Extract 3D patches
    DataPatch = array_to_patches(DataImage,
                                 patch_shape=(ps, ps, ps),
                                 extraction_step=stride,
                                 normalization=False)

    LabelPatch = array_to_patches(LabelImage,
                                  patch_shape=(ps, ps, ps),
                                  extraction_step=stride,
                                  normalization=False)

    HDF5Datas.append(DataPatch)
    HDF5Labels.append(LabelPatch)

    HDF5Datas = np.asarray(HDF5Datas)
    HDF5Datas = HDF5Datas.reshape(-1, ps, ps, ps)
    HDF5Labels = np.asarray(HDF5Labels).reshape(-1, ps, ps, ps)

    # Add channel axis !
    HDF5Datas = HDF5Datas[:, :, :, :, np.newaxis]
    HDF5Labels = HDF5Labels[:, :, :, :, np.newaxis]

    # Rearrange
    np.random.seed(0)  # makes the random numbers predictable
    RandomOrder = np.random.permutation(HDF5Datas.shape[0])
    HDF5Datas = HDF5Datas[RandomOrder, :, :, :, :]
    HDF5Labels = HDF5Labels[RandomOrder, :, :, :, :]

    # samples = 162
    # HDF5Datas = HDF5Datas[:samples, :, :, :, :]
    # HDF5Labels = HDF5Labels[:samples, :, :, :, :]

    hdf5name = "1010.h5"
    StartLocation = {'dat': (0, 0, 0, 0, 0), 'lab': (0, 0, 0, 0, 0)}
    CurrentDataLocation = store2hdf53D(filename=hdf5name,
                                       datas=HDF5Datas,
                                       labels=HDF5Labels,
                                       startloc=StartLocation,
                                       chunksz=64)

    launch_training(hdf5name, depth=10, nb_filters=64, kernel_size=3, padding=1)





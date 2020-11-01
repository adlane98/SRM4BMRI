import numpy as np
from tensorflow import pad
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.layers import Conv3D, ReLU, Input, Add, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from adamLRM import AdamLRM

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


if __name__ == '__main__':
    ps = 21
    model = SRReCNN3D([ps, ps, ps, 1], depth=10, nb_filters=64, kernel_size=3, padding=1)
    model.compile(optimizer=AdamLRM(learning_rate=0.0001), loss="mse")

    x1 = np.random.random((ps, ps, ps, 1))
    x2 = np.random.random((ps, ps, ps, 1))
    y1 = np.random.random((ps**3))
    y2 = np.random.random((ps**3))

    model.fit(x=np.array([x1, x2]), y=np.array([y1, y2]))

    g = model.get_weights()
    print(g)
#import tensorflow as tf
import numpy as np
import pdb
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow_datasets as tfds
from PIL import Image
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from numpy.random import seed
seed(1453)

#tf.random.set_seed(1471) #MAJ 


ds, info = tfds.load('mnist', split='train', shuffle_files=True,with_info = True)
#assert isinstance(ds, tf.data.Dataset)
print(ds)

dsp = ds.take(4)  # Only take a single example
print(dsp)
df = tfds.as_dataframe(ds.take(4),info)
e = df['image'][1]
e = e.reshape((28,28))
print(e.shape)
img_hr = Image.fromarray(e)
img_lr = img_hr.resize((14,14))
plt.figure()
plt.imshow(img_hr,cmap='gray')
plt.figure()
plt.imshow(img_lr,cmap='gray')
plt.show()

print("test")

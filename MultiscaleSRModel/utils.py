from datetime import datetime

import h5py
import numpy as np


def get_time():
    return datetime.now().strftime('%Y%m%d-%H%M%S')


def read_hdf5_file(hdf5_name_file):
    with h5py.File(hdf5_name_file) as hdf5data:
        data = list(hdf5data[list(hdf5data.keys())[0]])
        labels = list(hdf5data[list(hdf5data.keys())[1]])

    return data, labels


def read_hdf5_files(source_file):
    with open(source_file) as sf:
        hdf5_name_files = sf.readlines()
        data = []
        labels = []
        for hdf5_name_file in hdf5_name_files:
            with h5py.File(hdf5_name_file[:-1]) as hdf5data:
                data.extend(list(hdf5data[list(hdf5data.keys())[0]]))
                labels.extend(list(hdf5data[list(hdf5data.keys())[1]]))

    return np.asarray(data), np.asarray(labels)

import json
import os
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
            with h5py.File(hdf5_name_file[:-1], mode="r") as hdf5data:
                data.extend(list(hdf5data[list(hdf5data.keys())[0]]))
                labels.extend(list(hdf5data[list(hdf5data.keys())[1]]))

    return np.asarray(data), np.asarray(labels)


def write_metadata(filename, metadata):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w") as json_file:
        json.dump(metadata, json_file, indent=4)


def get_data_path():
    with open("pathes.json", "r") as json_file:
        pathes = json.load(json_file)
    return pathes["data"]


def get_metadata_path():
    with open("pathes.json", "r") as json_file:
        pathes = json.load(json_file)
    return pathes["metadata"]


def get_hdf5_path():
    with open("pathes.json", "r") as json_file:
        pathes = json.load(json_file)
    return pathes["hdf5"]

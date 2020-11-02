from SegSRGAN.SegSRGAN_training import SegSrganTrain
from SegSRGAN.Function_for_application_test_python3 import SegSRGAN_test as SganTest
from SegSRGAN.utils.patches import create_lr_hr_label, create_patch_from_df_hr
from SegSRGAN.utils.ImageReader import NIFTIReader

import os
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WEIGHTS = "weigths"
DATA = "data"
TRAINING = os.path.join(DATA, "training")

def new_low_res_to_list_res_max(new_low_res):
    list_res_max = new_low_res
    for i in range(len(list_res_max)):

        if len(list_res_max[i]) != 3:
            raise AssertionError('Not support this resolution !')

    print("Initial resolution given " + str(list_res_max))

    if len(list_res_max) == 1:
        list_res_max.extend(list_res_max)

    print("the low resolution of images will be choosen randomly between " + str(list_res_max[0]) + " and " +
          str(list_res_max[1]))
    return list_res_max

def main():
    data_folder = os.path.join(DATA,"")
    my_csv_path = os.path.join(TRAINING, "donnee_train.csv")
    snapshot_folder = os.path.join(TRAINING, "snapshot_folder")
    dice_file = os.path.join(TRAINING, "dice.csv")
    mse_file = os.path.join(TRAINING, "mse.csv")
    folder_training_temp_data = os.path.join(TRAINING, "batchs")
    contrast_max = 0.5
    percent_val_max = 0.03
    new_low_res = [(0.5, 0.5, 2), (0.5, 0.5, 3)]
    list_res_max = new_low_res_to_list_res_max(new_low_res)
    
    seg_sr_gan_train = SegSrganTrain(base_path = data_folder,
                                     contrast_max = contrast_max,
                                     percent_val_max = percent_val_max,
                                     list_res_max = list_res_max,
                                     training_csv = my_csv_path,
                                     multi_gpu = True)
    
    seg_sr_gan_train.train(snapshot_folder = snapshot_folder,
                           dice_file = dice_file,
                           mse_file = mse_file,
                           folder_training_data = folder_training_temp_data,
                           patch_size = 64,
                           training_epoch = 1)
    
if __name__ == "__main__":
    main()
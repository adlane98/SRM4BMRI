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

# Numéro tranche
T = 0.5
A = 1

def extract_file(file):
    nii_file = NIFTIReader(file)
    data = nii_file.get_np_array()
    resolution = nii_file.get_resolution()
    return nii_file, data, resolution


    
def get_patch_of_one(img_data, axis, number):
    if axis==0:
        return np.expand_dims(img_data[number,:,:], axis=axis)
    elif axis==1:
        return np.expand_dims(img_data[:,number,:], axis=axis)
    elif axis==2:
        return np.expand_dims(img_data[:,:,number], axis=axis)
    else:
        raise Exception( "axis compris entre 0 et 2")

def main():
    pass
    # nn_weights = os.path.join(WEIGHTS, "Perso_without_data_augmentation")

    
    # df = pd.DataFrame([[nii_img, nii_seg_img]], columns = ["HR_image", "Label_image"])
    
    # perc_val_max = 0.03
    # dossier_resultats = os.path.join(DATA,"resultats")
    # batch_size = 8
    # patch_size = 128
    # contrast_list = np.linspace(1 - 0.5, 1 + 0.5, df.shape[0])
    # lin_res_x = np.linspace(0.5, 0.5, df.shape[0])
    # lin_res_y = np.linspace(0.5, 0.5, df.shape[0])
    # lin_res_z = np.linspace(2, 3, df.shape[0])

    # res_test = [(lin_res_x[i],
    #                 lin_res_y[i],
    #                 lin_res_z[i]) for i in range(df.shape[0])]
    
    # path_save_npy, path_data_mini_batch, path_labels_mini_batch, remaining_patch = create_patch_from_df_hr(df, perc_val_max, dossier_resultats, batch_size, contrast_list, res_test, patch_size)
    
    # print(path_save_npy, path_data_mini_batch, path_labels_mini_batch, remaining_patch)



    # test_labels = np.load(path_labels_mini_batch[5])
    # test_datas = np.load(path_data_mini_batch[5])[:, 0, :, :, :][:, np.newaxis, :, :, :]

    # test_data = test_datas[0,0,:,:,:]
    # list_data = [test_datas[i,0,:,:,:] for i in range(len(test_datas))]
    # lr = test_datas[0,0,:,:,:]
    # hr = test_labels[0,0,:,:,:]
    # comparer_plot([hr, lr], 1, 0.5)



    # visualiser(hr, 1, T)
    # visualiser(seg, 1, 100)

    # lr, hr, label, up_scale, ori_lr = create_lr_hr_label(nii_img, nii_seg_img, new_resolution = (0.5, 1.5, 1.5), interp = "sitk")
    # print(up_scale)
    # visualiser(lr, 1, 100)
    # visualiser(hr, 0, 100)
    # visualiser(label, 0, 100)
    # print(ori_lr[:])
    # comparer_plot([hr, lr], A, T)
    
    # x, y, z = lr.shape
    # sgan_test = SganTest(nn_weights, x, y, z, True, False, True, 16, 32, 0)
    # esti_hr, esti_seg = sgan_test.test_by_patch(lr, step=1, by_batch=False)
    # # print(type(esti_hr), type(esti_seg))
    # # print(esti_hr)
    # comparer_plot([lr, esti_hr, hr], A, T)
    # comparer_plot([esti_seg, seg], A, T)
    # visualiser(esti_seg, 1, 100)


if __name__ == '__main__':
    main()
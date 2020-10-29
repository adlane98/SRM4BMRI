from SegSRGAN.Function_for_application_test_python3 import SegSRGAN_test as SganTest
from SegSRGAN.utils.patches import create_lr_hr_label
from SegSRGAN.utils.ImageReader import NIFTIReader

import os
import nibabel as nib
from matplotlib import pyplot as plt
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

WEIGHTS = "weigths"
DATA = "data"

# Numéro tranche
T = 150

def extract_file(file):
    nii_file = NIFTIReader(file)
    data = nii_file.get_np_array()
    resolution = nii_file.get_resolution()
    return nii_file, data, resolution

def visualiser(img_data, axis, number):
    print(img_data.shape)
    if axis==0:
        plt.imshow(img_data[number,:,:])
    elif axis==1:
        plt.imshow(img_data[:,number,:])
    elif axis==2:
        plt.imshow(img_data[:,:,number])
    else:
        raise Exception( "axis compris entre 0 et 2")
    plt.show()
    
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
    nn_weights = os.path.join(WEIGHTS, "Perso_without_data_augmentation")
    nii_seg_img = os.path.join(DATA, "img1\\ard.nii.gz")
    nii_img = os.path.join(DATA, "img1\\fit.nii.gz")
    
    nii_hr, hr, res_hr = extract_file(nii_img)
    nii_seg, seg, res_seg = extract_file(nii_seg_img)
    print("Résolution : ", res_hr)
    visualiser(hr, 1, T)
    # visualiser(seg, 1, 100)

    lr, hr, label, up_scale, ori_lr = create_lr_hr_label(nii_img, nii_seg_img, new_resolution = (0.9, 0.9, 0.9), interp = "sitk")
    # print(up_scale)
    # visualiser(lr, 1, 100)
    # visualiser(hr, 0, 100)
    # visualiser(label, 0, 100)
    # print(ori_lr[:])

    sgan_test = SganTest(nn_weights, 64, 64, 64, True, True, True, True, True, 1.2, fit_mask=False)
    esti_hr, esti_seg = sgan_test.test_by_patch(lr, step=128, by_batch=False)
    # print(type(esti_hr), type(esti_seg))
    # print(esti_hr)
    visualiser(esti_hr, 1, T)
    # visualiser(esti_seg, 1, 100)

if __name__ == '__main__':
    main()
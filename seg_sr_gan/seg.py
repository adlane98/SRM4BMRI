import os
import sys
from pathlib import Path
import importlib

from SegSRGAN.Function_for_application_test_python3 import segmentation



def main():

    wpath_base = importlib.util.find_spec("SegSRGAN").submodule_search_locations[0]
    wpath = os.path.join(wpath_base, "weights/Perso_without_data_augmentation")


    parent = "data"
    input_nii = os.path.join(parent, "img1\\fit.nii.gz")
    output_cortex_nii = os.path.join(parent, "cortex.nii.gz")
    output_sr_nii = os.path.join(parent, "SR.nii.gz")

    segmentation( input_file_path = input_nii,
                  step = 100,
                  new_resolution = (0.5,0.5,0.5),
                  patch = 128,
                  path_output_cortex = output_cortex_nii,
                  path_output_hr = output_sr_nii,
                  weights_path = wpath,
                  interpolation_type = "Lanczos")
                  
if __name__ == "__main__":
    main()
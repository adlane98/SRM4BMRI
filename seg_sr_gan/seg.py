from os.path import join, basename, normpath, isdir
from os import mkdir
import sys
from pathlib import Path
import importlib
from glob import glob

from SegSRGAN.Function_for_application_test_python3 import segmentation

DATA = "data"

def main():
    
    if len(sys.argv) < 1 + 2 or len(sys.argv) > 4:
        raise Exception ("We need the path of the niigz files repository, the path of outputs repo, and (optionnal) weight index")

    niigz_path = sys.argv[1]
    output_path = sys.argv[2]
    
    weight_name = "Perso_without_data_augmentation"
    if len(sys.argv) == 4:
        weight_name = sys.argv[3]

    wpath_base = importlib.util.find_spec("SegSRGAN").submodule_search_locations[0]
    wpath = normpath(join(wpath_base, "weights//"+weight_name))

    print("Weight : "+wpath)
    print("Output rep : "+join(DATA, output_path))
    
    if not isdir(join(DATA, output_path)):
        mkdir(join(DATA, output_path))
    
    files = glob(join(DATA, niigz_path)+'/*.nii.gz')

    for filepath in files:
        print("working on : "+ filepath)
        input_nii = filepath
        output_cortex_nii = join(join(DATA, output_path), "out_"+basename(filepath)[:-7]+"_cortex.nii.gz")
        output_sr_nii = join(join(DATA, output_path), "out_"+basename(filepath)[:-7]+"_sr.nii.gz")
        print("output : "+output_cortex_nii+" & "+output_sr_nii)
        segmentation( input_file_path = input_nii,
                    step = 100,
                    new_resolution = (0.5, 0.5, 0.5),
                    patch = 128,
                    path_output_cortex = output_cortex_nii,
                    path_output_hr = output_sr_nii,
                    weights_path = wpath,
                    interpolation_type = "Lanczos")
                  
if __name__ == "__main__":
    main()
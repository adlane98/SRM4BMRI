from os.path import join, basename, normpath, isdir
from os import mkdir
import sys
from pathlib import Path
import importlib
from glob import glob
from SegSRGAN.Function_for_application_test_python3 import segmentation


DATA = "data"

def main():
    """ run this script to all nii.gz file in a repository in input of the SegSRGAN given a weights file.
    first parameter : Path to the niigz files repository
    second parameter : Path to the outputs repository (if it doesnt exist, it is create)
    third parameter (optional) : weights file path (by default it is the Weights file without data augmentation from SegSRGAN's authors)

    Raises:
        Exception: Bad parameters
    """
    
    if len(sys.argv) < 1 + 2 or len(sys.argv) > 4:
        raise Exception ("We need the path of the niigz files repository, the path of outputs repo, and (optionnal) weight index")

    niigz_path = sys.argv[1]
    output_path = sys.argv[2]
    
    weight_name = "Perso_without_data_augmentation"
    if len(sys.argv) == 4:
        weight_name = sys.argv[3]

    wpath = normpath(join("weights", weight_name))

    print("Weight : "+wpath)
    print("Output rep : "+join(DATA, output_path))
    print("Input rep : "+join(DATA, niigz_path))
    
    if not isdir(join(DATA, output_path)):
        mkdir(join(DATA, output_path))
    
    files = glob(join(DATA, niigz_path)+'/*.nii.gz')
    if files == [] :
        print("Aucun fichier nii.gz détecté dans le dossier : "+join(DATA, niigz_path))
    
    for filepath in files:
        print("working on : "+ filepath)
        input_nii = filepath
        output_cortex_nii = join(join(DATA, output_path), "out_"+basename(filepath)[:-7]+"_cortex.nii.gz")
        output_sr_nii = join(join(DATA, output_path), "out_"+basename(filepath)[:-7]+"_sr.nii.gz")
        print("output : "+output_cortex_nii+" & "+output_sr_nii)
        segmentation( input_file_path = input_nii,
                    step = 50,
                    new_resolution = (0.5, 0.5, 0.5),
                    patch = 64,
                    path_output_cortex = output_cortex_nii,
                    path_output_hr = output_sr_nii,
                    weights_path = wpath,
                    interpolation_type = "Lanczos")
                  
if __name__ == "__main__":
    main()
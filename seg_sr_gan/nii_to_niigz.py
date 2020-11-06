import nibabel as nib
import sys
import nrrd
from nrrd_to_niigz import nrrd_to_niigz
import os
from glob import glob
import numpy as np

def main():
    """ convert nii file in niigz file
    give the nii files repository path in argument
    it will convert all nrrd files and output it in the same repository
    """
    if len(sys.argv) != 2:
        raise Exception("Give the repository of nii files")
    
    path = sys.argv[1]
    if not os.path.isdir(path):
        raise Exception("we need the path of a directory (a repo)")
    print("path : "+str(path))
    files = glob(path+'/*.nii')
    print("files : "+str(files))
    
    for filepath in files:
        file = nib.load(filepath)
        new_path = os.path.join(path, os.path.basename(filepath)[:-4]+".nii.gz")
        print("Sauvegarde - "+new_path)
        nib.save(file, new_path)

if __name__ == "__main__":
    main()
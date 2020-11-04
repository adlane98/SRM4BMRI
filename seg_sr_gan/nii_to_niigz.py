import nibabel as nib
import sys
import nrrd
from nrrd_to_niigz import nrrd_to_niigz
import os
from glob import glob
import numpy as np

def main():
    if len(sys.argv) != 2:
        raise Exception("Give the repository of nii files")
    
    path = sys.argv[1]
    if not os.path.isdir(path):
        raise Exception("we need the path of a directory (a repo)")
    
    files = glob(path+'/*.nii')
    
    for filepath in files:
        file = nib.load(filepath)
        nib.save(file, os.path.join(path, os.path.basename(filepath)[:-4]+".nii.gz"))

if __name__ == "__main__":
    main()
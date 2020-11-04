import os
from glob import glob
import nrrd 
import nibabel as nib
import numpy as np
import sys

def nrrd_to_niigz(path, files):
    for file in files:
        _nrrd = nrrd.read(file)
        data = _nrrd[0]
        
        img = nib.Nifti1Image(data, np.eye(4))
        nib.save(img, os.path.join(path, os.path.basename(file)[:-5]+'.nii.gz'))

def main():
    if len(sys.argv) != 2:
        raise Exception("need the repository's path")
    
    path = sys.argv[1]
    if not os.path.isdir(path):
        raise Exception("we need the path of a directory (a repo)")
    
    files = glob(path+'/*.nrrd')

    nrrd_to_niigz(path, files)

if __name__ == "__main__":
    main()
    
# We recommend to use a virtualenv
#
# $ virtualenv my_test
# $ source ./my_test/bin/active
# $ pip install SegSRGAN

import SegSRGAN
from nilearn import plotting
import nibabel as nib
import vtk
from os.path import join
from matplotlib import pyplot as plt
data = "data"


def main():
    file = join(data, "1010.nii")
    img = nib.load(file)
    img_data = img.get_data()
    fig=plt.figure()
    for i in range(62):
        fig.add_subplot(1,3,1)
        plt.imshow(img_data[i,:,:])
        fig.add_subplot(1,3,2)
        plt.imshow(img_data[:,i,:])
        fig.add_subplot(1,3,3)
        plt.imshow(img_data[:,:,i])
        plt.pause(0.01)
if __name__ == "__main__":
    main()
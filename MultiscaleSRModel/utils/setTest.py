import nibabel as nib
import numpy as np
import scipy.ndimage

import SimpleITK as sitk

from utils.utils3D import modcrop3D, imadjust3D


if __name__ == '__main__':
    sg = 1
    scale = (2, 2, 2)
    #border = (3, 3, 1)
    order = 3

    reference = ["1010.nii","1037.nii","1380.nii"]


    for ref in reference :
        ReferenceNifti = sitk.ReadImage(ref)

        # Get data from NIFTI
        ReferenceImage = np.swapaxes( sitk.GetArrayFromImage(ReferenceNifti), 0, 2).astype('float32')

        # Normalization
        ReferenceImage = imadjust3D(ReferenceImage, [0, 1])

        # ===== Generate input LR image =====
        # Blurring
        BlurReferenceImage = scipy.ndimage.filters.gaussian_filter(ReferenceImage,
                                                                   sigma=sg)


        # Modcrop to scale factor
        BlurReferenceImage = modcrop3D(BlurReferenceImage, scale)
        #ReferenceImage = modcrop3D(ReferenceImage, scale)

        # Downsampling
        LowResolutionImage = scipy.ndimage.zoom(BlurReferenceImage,
                                                zoom=(1 / float(idxScale) for idxScale in scale),
                                                order=order)

        # Cubic Interpolation
        InterpolatedImage = scipy.ndimage.zoom(LowResolutionImage,
                                               zoom=scale,
                                               order=order)
        new_image = nib.Nifti1Image(InterpolatedImage, affine=np.eye(4))
        nib.nifti1.save(new_image, "./data/lowImage_"+ref)
        # Shave border
        #LabelImage = shave3D(ReferenceImage, border)
        #DataImage = shave3D(InterpolatedImage, border)


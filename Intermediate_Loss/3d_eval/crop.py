from scipy.ndimage import zoom
import numpy as np
import nibabel as nib


def modcrop3D(img, modulo):
    import math
    img = img[0:int(img.shape[0] - math.fmod(img.shape[0], modulo[0])),
          0:int(img.shape[1] - math.fmod(img.shape[1], modulo[1])),
          0:int(img.shape[2] - math.fmod(img.shape[2], modulo[2]))]
    return img


def imadjust3D(image, newRange=None):
    Min = np.min(image)
    Max = np.max(image)
    newMin = newRange[0]
    newMax = newRange[1]
    temp = (newMax - newMin) / float(Max - Min)
    image = ((image - Min) * temp + newMin)
    return image

reference_nifti = nib.load(r'C:\Users\Alexandre\Repertoire\SRM4BMRI\Intermediate_Loss\test_mri\1010.nii')
downsampling_scale = [2,2,2]
reference_image = reference_nifti.get_data()
affine = reference_nifti.affine
#reference_image = imadjust3D(reference_image, [0, 1])
reference_image = modcrop3D(reference_image, downsampling_scale)

low_resolution_image = zoom(
    reference_image,
    zoom=(1 / float(idxScale) for idxScale in downsampling_scale),
    order=3
)

new_image = nib.Nifti1Image(
    low_resolution_image.astype(np.float64), affine=affine
 )
nib.save(new_image, "downsampled_1010.nii")
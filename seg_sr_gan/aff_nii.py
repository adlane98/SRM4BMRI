from SegSRGAN.utils.ImageReader import NIFTIReader
from nilearn import plotting
import nibabel as nib
from os.path import join
from matplotlib import pyplot as plt
import numpy as np

DATA = "data"

def extract_nii_file(file:str):
    """ read a nii file

    Args:
        file (str): [filepath]

    Returns:
        [tuple(nii_file, numpy_array, float)]: [a tuple with the nii file class, the 3d array of the image, and the resolution]
    """
    nii_file = NIFTIReader(file)
    data = nii_file.get_np_array()
    resolution = nii_file.get_resolution()
    return nii_file, data, resolution

def visualiser(img_data : np.ndarray, axis : int, number : float, show : bool = True, *args, **kwargs):
    """Visualisation d'une image 3d (qui provient d'un .nii)

    Args:
        img_data ([np.array]): [le tableau numpy qui provient de l'image nii]
        axis ([int]): [le numéro de l'axe (de 0 à 2)]
        number ([float]): [le pourcentage correspondant à la position de la tranche à afficher]
        show (bool, optional): [Si on veut que ça s'affichet par un plt.show()]. Defaults to True.

    Raises:
        Exception: [si l'axes n'est pas compris en 0 et 2]
    """
    x,y,z = img_data.shape
    if number > 1 or number < 0 : 
        raise Exception ("number est un pourcentage")
    if axis==0:
        plt.imshow(img_data[int(x*number),:,:], *args, **kwargs)
    elif axis==1:
        plt.imshow(img_data[:,int(y*number),:], *args, **kwargs)
    elif axis==2:
        plt.imshow(img_data[:,:,int(z*number)], *args, **kwargs)
    else:
        raise Exception( "axis compris entre 0 et 2")
    if show == True:
        plt.show()
    
def visualiser_n_img(list_img : list, axis:int, number:float, *args, **kwargs):
    """Visualisation simultanée d'image 3d provenant de .nii

    Args:
        list_img (list<nd.array>): [list des images 3d .nii sous tableau numpy]
        axis ([int]): [le numéro de l'axe (de 0 à 2)]
        number ([float]): [le pourcentage correspondant à la position de la tranche à afficher]
    """

    fig = plt.figure(1)
    n_img = len(list_img)
    for i, img in enumerate(list_img):
        fig.add_subplot(1, n_img, i+1)
        visualiser(img, axis, number, show=False, *args, **kwargs)
    plt.show()

def main():
        
    seg_out = join(DATA, "cortex.nii.gz")
    sr_out = join(DATA, "SR.nii.gz")
    hr_in = join(DATA, "img3\\010.nii.gz")
    #seg_in = join(DATA, "img1\\ard.nii.gz")
    
    seg_out, arr_seg_out, res_seg_out = extract_nii_file(seg_out)
    sr_out, arr_sr_out, res_sr_out = extract_nii_file(sr_out)
    hr_in, arr_hr_in, res_hr_in = extract_nii_file(hr_in)
    #seg_in, arr_seg_in, res_seg_in = extract_nii_file(seg_in)
    
    print("resolution hr in : ", str(res_hr_in), " resolution sr out : ", str(res_sr_out))
    print("resolution seg in : ", str("x"), " resolution seg out : ", str(res_seg_out))
    visualiser_n_img([arr_hr_in, arr_sr_out, arr_seg_out], 1, 0.5, cmap="gray")

if __name__ == "__main__":
    main()
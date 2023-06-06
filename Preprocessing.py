# iterative version, over a set of volumes, of the rescaling and resizing of the volumes
# note that you should change the paths to directories based on where you saved the original database and where you intend to save the preprocessed images, and the number of cases you want to preprocess

# required libraries
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import skimage as sk
from tqdm import tqdm
import os

# paths
dataset_folder = "D:\Programming\BioCV\kits19\data\\"
preprocessed_folder = "D:\Programming\BioCV\Project\PreprocessedImages\\"
if not os.path.isdir(preprocessed_folder+"Images2D\\"):
    os.makedirs(preprocessed_folder+"Images2D\\")
if not os.path.isdir(preprocessed_folder+"Segmentations2D\\"):
    os.makedirs(preprocessed_folder+"Segmentations2D\\")

# function to load images
def img_load(case):
    s_case = str(case)
    path = dataset_folder+"case_"+s_case.zfill(5)+"\imaging.nii.gz"
    mri = nib.load(path)
    mri_img = mri.get_fdata()
    return_array = [mri, mri_img]
    return return_array

# function to load segmentations
def segment_load(case):
    s_case = str(case)
    path = dataset_folder+"case_"+s_case.zfill(5)+"\segmentation.nii.gz"
    mri = nib.load(path)
    mri_img = mri.get_fdata()
    return_array = [mri, mri_img]
    return return_array

# function to calculate the voxel dimention for the volume
def vox_dim_f(mri):
    pixdim = mri.header['pixdim']
    z_space = pixdim[1]
    x_space = pixdim[2]
    y_space = pixdim[3]
    vox_dim = (x_space, y_space, z_space)
    return vox_dim

# function to rescale the image to reach a certain resolution
def my_rescale(img, t_res, v_dim):
    scale_vector = (v_dim[0]/t_res[0], v_dim[1]/t_res[1], v_dim[2]/t_res[2])
    isotr_img = sk.transform.rescale(img, scale_vector, order=None, preserve_range=True,  mode='constant')
    return isotr_img

# function to resize the image to a certain shape
def my_resize(is_img, t_shape):
    isotr_img_shape = is_img.shape
    factors = (t_shape[0]/isotr_img_shape[0], t_shape[1]/isotr_img_shape[1], t_shape[2]/isotr_img_shape[2])
    isotr_reshaped = sk.transform.resize(is_img, t_shape, order = 0)
    return isotr_reshaped

# functions that groups all the preprocessing phases for the images
def my_process_img(case):
    mri = img_load(case)
    mri_data = mri[0]
    img = mri[1]
    vox_dim = vox_dim_f(mri_data)
    isotropic_img = my_rescale(img, [1,1,1],vox_dim)
    reshaped_isotr_img = my_resize(isotropic_img, [256,256,256])
    return reshaped_isotr_img

# functions that groups all the preprocessing phases for the segmentations
def my_process_segm(case):
    mri = segment_load(case)
    mri_data = mri[0]
    img = mri[1]
    vox_dim = vox_dim_f(mri_data)
    reshaped_isotr_img = my_resize(img, [256,256,256])
    return reshaped_isotr_img

# function to save the preprocessed images and segmentations
def my_save(case, processed_file, i):
    s_case = str(case)
    type = ["Images2D\\", "Segmentations2D\\"]
    file = (processed_file/255).astype(np.float16)
    for slice in range(0,256):
        s_slice = str(slice)
        img2d = file[slice,:,:]
        path_save = preprocessed_folder+type[i]+s_case+"_"+s_slice+".npy"
        np.save(path_save, img2d)

first_case = 0
last_case = 1
pbar = tqdm(total=100)
for case in range(first_case,last_case):
    files = my_process_img(case)
    my_save(case, files, 0)
    if case < 210:
        files = my_process_segm(case)
        my_save(case, files, 1)
    pbar.update(100/(last_case-first_case))
pbar.close()
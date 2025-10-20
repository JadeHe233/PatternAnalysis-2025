import numpy as np
import nibabel as nib
from torch.utils.data import Dataset
import torch
import torchio as tio

# Helper functions
def to_channels(arr: np.ndarray, dtype=np.uint8)-> np.ndarray:
    channels = np.unique(arr)
    res = np.zeros(arr.shape + ( len(channels),), dtype=dtype)
    for c in channels:
        c = int(c)
        res[..., c:c+1][arr == c] = 1
    
    return res

def load_data_3D(imageNames, normImage=False, categorical=False,
                dtype=np.float32, getAffines=False,
                early_stop=False):
    '''
    Load medical image data from names, cases list provided into a list for each.

    This function pre-allocates 5D arrays for conv3d to avoid excessive memory usage.

    normImage: bool (normalise the image 0.0-1.0)
    orient: Apply orientation and resample image? Good for images with large slice
            thickness or anisotropic resolution
    dtype: Type of the data. If dtype=np.uint8, it is assumed that the data is labels
    early_stop: Stop loading pre-maturely? Leaves arrays mostly empty, for quick
                loading and testing scripts.
    '''
    affines = []

    #~ interp = 'continuous'
    interp = 'linear'
    if dtype == np.uint8: #assume labels
        interp = 'nearest'

    #get fixed size
    num = len(imageNames)
    niftiImage = nib.load(imageNames[0])
    first_case = niftiImage.get_fdata(caching='unchanged')
    if len(first_case.shape) == 4:
        first_case = first_case[:, :, :, 0] #sometimes extra dims, remove
    if categorical:
        first_case = to_channels(first_case, dtype=dtype)
        rows, cols, depth, channels = first_case.shape
        images = np.zeros((num, rows, cols, depth, channels), dtype=dtype)
    else:
        rows, cols, depth = first_case.shape
        images = np.zeros((num, rows, cols, depth), dtype=dtype)

    for i, inName in enumerate(imageNames):
        niftiImage = nib.load(inName)
        inImage = niftiImage.get_fdata(caching='unchanged') #read disk only
        affine = niftiImage.affine
        if len(inImage.shape) == 4:
            inImage = inImage[:, :, :, 0] #sometimes extra dims in HipMRI_study data
        inImage = inImage[:, :, :depth] # clip slices
        inImage = inImage.astype(dtype)
        if normImage:
            #~ inImage = inImage / np.linalg.norm(inImage)
            #~ inImage = 255. * inImage / inImage.max()
            inImage = (inImage - inImage.mean()) / inImage.std()
        if categorical:
            inImage = to_channels(inImage, dtype=dtype)
            #~ images[i,:,:,:,:] = inImage
            images[i,:inImage.shape[0],:inImage.shape[1],:inImage.shape[2],:inImage.shape[3]] = inImage #with pad
        else:
            #~ images[i,:,:,:] = inImage
            images[i,:inImage.shape[0],:inImage.shape[1],:inImage.shape[2]] = inImage #with pad
        
        affines.append(affine)
        if i > 20 and early_stop:
            break

    if getAffines:
        return images, affines
    else:
        return images

class MRIDataset(Dataset):
    def __init__(self, mri_files, label_files, transform=None):
       self.mri_files = mri_files
       self.label_files = label_files
       self.transform = transform

    def __len__(self):
        return len(self.mri_files)
    
    def __getitem__(self, idx):
        image = load_data_3D([self.mri_files[idx]], normImage=True)[0]
        label = load_data_3D([self.label_files[idx]], categorical=True, dtype=np.uint8)[0]
        
        # Crop data to prevent CUDA OOM
        image = image[32:160, 32:160, 32:96]
        label = label[32:160, 32:160, 32:96, :]

        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # [1, D, H, W]
        label = torch.tensor(label, dtype=torch.float32).permute(3, 0, 1, 2) # [C, D, H, W]

        # Apply the augmentation
        if self.transform:
            subject = tio.Subject(
                image=tio.ScalarImage(tensor=image),
                label=tio.LabelMap(tensor=label)
            )
            subject = self.transform(subject)
            image = subject.image.data
            label = subject.label.data

        return image, label
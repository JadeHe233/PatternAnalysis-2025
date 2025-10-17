from torch.utils.data import DataLoader, random_split
from dataset import MRIDataset
import glob
import torch

# Set the random seed
seed = 233
generator= torch.Generator().manual_seed(seed)

# Use glob to retrieve all nifti files
# Use sorted to match the MRIs with their labels
mri_files = sorted(glob.glob("HipMRI_study_complete_release_v1/semantic_MRs_anon/*.nii.gz"))
label_files = sorted(glob.glob("HipMRI_study_complete_release_v1/semantic_labels_anon/*.nii.gz"))

dataset = MRIDataset(mri_files, label_files)

# Split dataset 
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

# Data loaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set,   batch_size=1, shuffle=False)
test_loader = DataLoader(test_set,   batch_size=1, shuffle=False)
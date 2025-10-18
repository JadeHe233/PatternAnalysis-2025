from torch.utils.data import DataLoader, random_split
from dataset import MRIDataset
from modules import *
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import time

# Start a new wandb run to track this script.
run = wandb.init(
    entity="s4801815-the-university-of-queensland",
    project="prostate-mri-unet3d",
    # Track hyperparameters and run metadata.
    config={
        "learning_rate": 1e-4,
        "architecture": "Unet3D",
        "dataset": "HipMRI",
        "epochs": 50,
    },
)

# Set the random seed
seed = 233
generator= torch.Generator().manual_seed(seed)

mri_dir   = "HipMRI_study_complete_release_v1/semantic_MRs_anon"
label_dir = "HipMRI_study_complete_release_v1/semantic_labels_anon"

# Use glob to retrieve all nifti files
# Use sorted to match the MRIs with their labels
mri_files = sorted(glob.glob(f"{mri_dir}/*.nii.gz"))
label_files = sorted(glob.glob(f"{label_dir}/*.nii.gz"))

dataset = MRIDataset(mri_files, label_files)

# Split dataset 
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size  = len(dataset) - train_size - val_size

train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=generator)

# Data loaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set,   batch_size=1, shuffle=False, num_workers=4)
test_loader = DataLoader(test_set,   batch_size=1, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet3D(in_channels=1, out_channels=6).to(device)

num_epochs = 50
criterion = DiceLoss()
optimiser = optim.Adam(model.parameters(), lr=1e-4)

losses = []

import os
os.makedirs("checkpoints", exist_ok=True)

print("Start training...")
start_time = time.time()
for epoch in range(num_epochs):
    epoch_start = time.time()
    # --- Train ---
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
    
        optimiser.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimiser.step()

        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    losses.append(avg_train_loss)

    # --- Validate ---
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device) 
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{num_epochs} | "
      f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
      f"Time: {epoch_time:.2f} sec")

    # Log to wandb
    wandb.log({
        "train_loss": train_loss,
        "val_loss": val_loss,
        "epoch_time_sec": epoch_time,
        "epoch": epoch
    })

total_time = time.time() - start_time
print(f"✅ Training complete in {total_time/60:.2f} minutes.")
wandb.log({"total_training_time_min": total_time / 60})
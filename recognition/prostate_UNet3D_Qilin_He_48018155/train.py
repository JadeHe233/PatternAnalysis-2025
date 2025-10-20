from torch.utils.data import DataLoader, random_split
from dataset import MRIDataset
from modules import *
import os
import glob
import torch
import torch.optim as optim
import wandb
import time
import torchio as tio

print("Script started...")

# Start a new wandb run to track this script.
run = wandb.init(
    entity="s4801815-the-university-of-queensland",
    project="prostate-mri-unet3d",
    mode="offline",
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
print(f"  Found {len(mri_files)} MRI files and {len(label_files)} label files.")

# Augmentation for the training set
train_transform = tio.Compose([
    tio.RandomFlip(axes=('LR',), flip_probability=0.5), # Left-right flip
    tio.RandomAffine(scales=(0.9, 1.1), degrees=10), # Scale ±10%, rotate ±10°
    tio.RandomElasticDeformation(num_control_points=7, max_displacement=7),  # Tissue warping
    tio.RandomGamma(log_gamma=(-0.3, 0.3)), # Brightness/contrast change
    tio.RandomNoise(std=0.01), # Slight Gaussian noise
    tio.ZNormalization() # Normalise intensity
])

val_transform = tio.Compose([tio.ZNormalization()]) # Augmentation for the val/test set

# Split the dataset indices
dataset_size = len(mri_files)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size  = dataset_size - train_size - val_size
print(f"Split into {train_size} train / {val_size} val / {test_size} test")

train_subset, val_subset, test_subset = torch.utils.data.random_split(
    range(dataset_size),
    [train_size, val_size, test_size],
    generator=generator)

train_indices = train_subset.indices
val_indices = val_subset.indices
test_indices = test_subset.indices

# Save splits
torch.save({
    'train_indices': train_indices,
    'val_indices': val_indices,
    'test_indices': test_indices
}, "data_splits.pt")

# Split the actual dataset
print("Splitting the dataset...")
train_files = [mri_files[i] for i in train_indices]
train_labels = [label_files[i] for i in train_indices]

val_files = [mri_files[i] for i in val_indices]
val_labels = [label_files[i] for i in val_indices]

test_files = [mri_files[i] for i in test_indices]
test_labels = [label_files[i] for i in test_indices]

# Instanciate the split datasets
train_set = MRIDataset(train_files, train_labels, transform=train_transform)
val_set   = MRIDataset(val_files, val_labels, transform=val_transform)
test_set  = MRIDataset(test_files, test_labels, transform=val_transform)

# Data loaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
print("DataLoaders created successfully")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = UNet3D(in_channels=1, out_channels=6).to(device)
print("Model created.")

num_epochs = 50
criterion = DiceLoss()
optimiser = optim.Adam(model.parameters(), lr=1e-4)

# Track best validation loss
best_val_loss = float('inf')
best_model_path = None

# Early stopping parameters
patience = 10         # Number of epochs to wait for improvement
no_improve_epochs = 0 # Counter for epochs without improvement
early_stop = False

# Create a folder to save the checkpoints
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
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "epoch_time_sec": epoch_time,
        "epoch": epoch + 1
    })

    # Save the best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_path = f"checkpoints/unet3d_best.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimiser.state_dict(),
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': best_val_loss
        }, best_model_path)
        print(f"New best model saved at epoch {epoch+1}: {best_model_path}")
        no_improve_epochs = 0  # reset counter
    else:
        no_improve_epochs += 1
    
    # Check for early stopping
    if no_improve_epochs >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}")
        early_stop = True
        break

total_time = time.time() - start_time
if early_stop:
    print(f"Training stopped early after {epoch+1} epochs due to no improvement.")
else:
    print(f"Training completed all {num_epochs} epochs.")

print(f"Total training time: {total_time/60:.2f} minutes.")
wandb.log({
    "total_training_time_min": total_time / 60,
    "stopped_epoch": epoch + 1,
    "early_stopped": early_stop
})
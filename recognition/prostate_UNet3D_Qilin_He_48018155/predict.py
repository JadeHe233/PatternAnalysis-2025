import torch
from modules import UNet3D
from dataset import MRIDataset
from torch.utils.data import DataLoader
from modules import DiceLoss
import glob
import torchio as tio
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def dice_per_class(pred, target, num_classes=6, eps=1e-6):
    """
    Compute the Dice Similarity Coefficient for each class.

    Args:
        pred: Predictions made by the model in the shape of [B, C, D, H, W]. 
        B - batch size; C - number of channels; D - depth, number of slices along the z-axis
        H - height; W - width.
        target: Ground-truth labels of shape [B, D, H, W], where each voxel value corresponds to a class index.
        num_classes: the number of segmentation labels.
        eps: Small constant added to avoid division by zero.
    
    Returns:
        list: a list of Dice Similarity Coefficient for each class.
    """
    dice_scores = []
    pred = torch.argmax(pred, dim=1)  # [B, D, H, W]
    
    for c in range(num_classes):
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        dice = (2.0 * intersection + eps) / (union + eps)
        dice_scores.append(dice.item())
    return dice_scores

checkpoint_path = "checkpoints/unet3d_best.pth"
mri_dir   = "HipMRI_study_complete_release_v1/semantic_MRs_anon"
label_dir = "HipMRI_study_complete_release_v1/semantic_labels_anon"

# Load the split indices
splits = torch.load("data_splits.pt")
test_indices = splits["test_indices"]

mri_files = sorted(glob.glob(f"{mri_dir}/*.nii.gz"))
label_files = sorted(glob.glob(f"{label_dir}/*.nii.gz"))

test_files = [mri_files[i] for i in test_indices]
test_labels = [label_files[i] for i in test_indices]

val_transform = tio.Compose([tio.ZNormalization()]) # Augmentation for the val/test set

test_set  = MRIDataset(test_files, test_labels, transform=val_transform)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = UNet3D(in_channels=1, out_channels=6).to(device)

# Load the trained model
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"]) # Load the parameters
epoch = checkpoint["epoch"]
val_loss = checkpoint["avg_val_loss"]

print(f"Model loaded from epoch {epoch} (val_loss={val_loss:.4f})")

model.eval()
dice_all_classes = []

criterion = DiceLoss()
test_loss = 0

with torch.no_grad():
    for i, (images, labels) in enumerate(test_loader):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss, _ = criterion(outputs, labels)
        test_loss += loss.item()

        dice_scores = dice_per_class(outputs, torch.argmax(labels, dim=1))
        dice_all_classes.append(dice_scores)

avg_test_loss = test_loss / len(test_loader)
dice_all_classes = np.array(dice_all_classes)
mean_dice_per_class = dice_all_classes.mean(axis=0)
std_dice_per_class = dice_all_classes.std(axis=0)

print(f"Test Dice loss: {avg_test_loss:.4f}  →  Dice ≈ {1 - avg_test_loss:.4f}")

# --- Bar plot for Dice per Class with Std ---
"""
class_names = ["Background", "Body", "Bones", "Bladder", "Rectum", "Prostate"]
print("\n Mean Dice Similarity Coefficient per class:")
for i, (m, s) in enumerate(zip(mean_dice_per_class, std_dice_per_class)):
    print(f"  {class_names[i]:<10s}: {m:.4f} ± {s:.4f}")

plt.figure(figsize=(8, 5))
bars = plt.bar(
    range(6),
    mean_dice_per_class,
    yerr=std_dice_per_class,
    capsize=4,
    color=["gray", "lightblue", "gold", "lime", "orange", "red"],
)

# Add numeric value labels above bars
for i, bar in enumerate(bars):
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.03,
        f"{mean_dice_per_class[i]:.3f} ± {std_dice_per_class[i]:.3f}",
        ha="center", va="bottom", fontsize=9,
    )

plt.xticks(range(6), class_names, rotation=30)
plt.ylabel("Dice Similarity Coefficient")
plt.title("Mean Dice Coefficient per Class (Test Set)")
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle="--", alpha=0.6)

os.makedirs("Figures", exist_ok=True)
plt.savefig("Figures/dice_per_class_with_std.png", dpi=300, bbox_inches="tight")
plt.show()
"""

# --- Plot for Overlay Prediction vs Ground Truth ---
example_idx = 0
mri, label = test_set[example_idx]
mri = mri.unsqueeze(0).to(device)  # [1, 1, D, H, W]
label = label.unsqueeze(0).to(device)

with torch.no_grad():
    pred = model(mri)
    pred_softmax = torch.softmax(pred, dim=1)
    pred_label = torch.argmax(pred_softmax, dim=1).cpu().squeeze(0)  # [D, H, W]
    gt_label = torch.argmax(label, dim=1).cpu().squeeze(0)  # [D, H, W]
    mri = mri.cpu().squeeze().numpy()

# Pick a central slice
slice_idx = pred_label.shape[0] // 2

plt.figure(figsize=(12, 4))

# MRI Slice
plt.subplot(1, 3, 1)
plt.imshow(mri[slice_idx], cmap='gray')
plt.title("MRI Slice")
plt.axis('off')

# Ground Truth Label
plt.subplot(1, 3, 2)
plt.imshow(gt_label[slice_idx], cmap='tab10', vmin=0, vmax=5)
plt.title("Ground Truth Label")
plt.axis('off')

# Predicted Overlay
plt.subplot(1, 3, 3)
plt.imshow(mri[slice_idx], cmap='gray')
plt.imshow(pred_label[slice_idx], alpha=0.5, cmap='tab10', vmin=0, vmax=5)
plt.title("Predicted Label Overlay")
plt.axis('off')

# --- Add legend ---
# Class names (adjust if your labels differ)
class_labels = ["Background", "Body", "Bones", "Bladder", "Rectum", "Prostate"]

# Use the same colormap to get the corresponding colors
cmap = plt.get_cmap('tab10')
colors = [cmap(i) for i in range(len(class_labels))]

# Create legend handles
patches = [mpatches.Patch(color=colors[i], label=class_labels[i]) for i in range(len(class_labels))]
plt.legend(
    handles=patches,
    bbox_to_anchor=(1.05, 0.5),
    loc='center left',
    borderaxespad=0.,
    title="Classes"
)

plt.tight_layout()
os.makedirs("Figures", exist_ok=True)
plt.savefig("Figures/pred_vs_gt_overlay.png", dpi=300, bbox_inches='tight')
plt.show()
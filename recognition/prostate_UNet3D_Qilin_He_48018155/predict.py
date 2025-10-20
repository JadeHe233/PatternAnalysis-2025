import torch
from modules import UNet3D  # make sure you have the same model definition file

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNet3D(in_channels=1, out_channels=6).to(device)
checkpoint = torch.load("unet3d_best.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print(f"✅ Model loaded from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f})")

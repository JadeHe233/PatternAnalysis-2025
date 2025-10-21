import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found! Using CPU.")

# A basic 3D Conv block
class BasicConv3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_p),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout3d(dropout_p)
        )
    
    def forward(self, x):
        return self.block(x)
    
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=6, dropout_p=0.1):
        super().__init__()
        
        # Encoder (Downsampling)
        self.enc1 = BasicConv3D(in_channels, 32, dropout_p)
        self.enc2 = BasicConv3D(32, 64, dropout_p)
        self.enc3 = BasicConv3D(64, 128, dropout_p)
        
        # Bottleneck
        self.bottleneck = BasicConv3D(128, 256, dropout_p)

        # Decoder (Upsampling)
        self.up3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.dec3 = BasicConv3D(128 + 128, 128, dropout_p)

        self.up2 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.dec2 = BasicConv3D(64 + 64, 64, dropout_p)

        self.up1 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec1 = BasicConv3D(32 + 32, 32, dropout_p)

        # Final output
        self.out = nn.Conv3d(32, out_channels, kernel_size=1)

        # Downsample
        self.pool = nn.MaxPool3d(2)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x) # 1 -> 32
        e2 = self.enc2(self.pool(e1)) # 32 -> 64
        e3 = self.enc3(self.pool(e2)) # 64 -> 128
        b = self.bottleneck(self.pool(e3)) # 128 -> 256

        # Decoder with skip connections
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1)) # 128 + 128 -> 128
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1)) # 64 + 64 -> 64
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1)) # 32 + 32 -> 32

        return self.out(d1) # 32 -> 6

class DiceLoss(nn.Module):
    """
    Adapted from: https://colab.research.google.com/drive/1VOsZSyRhyuHLmgoqGriQk01ub4bKNmZ1?usp=sharing
    """
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        Args:
            predictions: raw logits output by the model with the shape of [B, C, D, H, W]
            targets: one-hot encoded labels with the shape of [B, C, D, H, W]
        """
        predictions = F.softmax(predictions, dim=1) # Convert the logits into probabilities
        
        predictions = predictions.flatten(2) # [B, C, N]
        targets = targets.flatten(2) # [B, C, N]

        intersection = (predictions * targets).sum(dim=2)
        dice_coeff = (2.0 * intersection + self.smooth) / (predictions.sum(dim=2) + targets.sum(dim=2) + self.smooth)
        
        loss_per_class = 1 - dice_coeff.mean(dim=0)  # [C]
        loss = loss_per_class.mean()  # scalar mean

        return loss, loss_per_class
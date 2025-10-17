import torch
import torch.nn as nn
import torchvision.transforms as transforms
import time

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print("Warning CUDA not found! Using CPU.")

# Hyper-parameters 
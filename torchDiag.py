#!/usr/bin/env python3
import torch

# Check if CUDA (NVIDIA GPUs) is available
print("Is CUDA available?", torch.cuda.is_available())

# Display the number of GPUs
print("Number of GPUs available:", torch.cuda.device_count())

# List each GPU's name
for i in range(torch.cuda.device_count()):
    print(f"GPU {i} Name:", torch.cuda.get_device_name(i))

# Check if MPS (Metal Performance Shaders for macOS) is available
if hasattr(torch.backends, 'mps'):
    print("Is MPS available?", torch.backends.mps.is_available())

# Check if CPU is supported (always True for PyTorch)
print("Is CPU available?", True)


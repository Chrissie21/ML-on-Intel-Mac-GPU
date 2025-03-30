import torch
device = torch.device("mps")  # Use Metal GPU
print("Using device:", device)

# Test a simple tensor operation
x = torch.rand((3, 3)).to(device)
print(x)

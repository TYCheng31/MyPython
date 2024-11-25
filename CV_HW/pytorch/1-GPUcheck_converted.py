import torch

# 檢查 GPU 是否可用
gpu_available = torch.cuda.is_available()
print("GPU Available:", gpu_available)

if gpu_available:
    current_device = torch.cuda.current_device()
    device_name = torch.cuda.get_device_name(current_device)
    print("Current Device Index:", current_device)
    print("Device Name:", device_name)
else:
    print("No GPU available.")

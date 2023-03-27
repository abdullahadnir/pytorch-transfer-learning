import torch
print("Pytorch CUDA Version is ", torch.version.cuda)

# setting device on GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print()
#Additional Info when using cuda
if device.type == "cuda":
 print(torch.cuda.get_device_name(0))
 print("Memory Usage:")
 print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,1), "GB")
 print("Cached: ", round(torch.cuda.memory_reserved(0)/1024**3,1),"GB")
if torch.cuda.is_available():
    device=torch.device("cuda:0")
    print("Training on GPU... Ready for HyperJump...")
else:
    device = torch.device("cpu")
#     print("Training on CPU... May the force be with you...")

print(torch.cuda.is_available())


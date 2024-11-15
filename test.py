import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

if device == "cuda":
    print(f"Current device number: {torch.cuda.current_device()}")
    print(f"GPU device name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"Number of available GPUs: {torch.cuda.device_count()}")


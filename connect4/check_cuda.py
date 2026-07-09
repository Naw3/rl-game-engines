"""Quick sanity check after CUDA setup."""
import torch
import onnxruntime
import pygame
import onnx
import nvidia.cudnn  # noqa: F401  -- just check it imports

print(f"torch:           {torch.__version__}")
print(f"torch CUDA:      {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"torch device:    {torch.cuda.get_device_name(0)}")
    print(f"torch CUDA ver:  {torch.version.cuda}")
print(f"onnxruntime:     {onnxruntime.__version__}")
print(f"nvidia.cudnn:    imported OK")
print(f"pygame:          {pygame.__version__}")
print(f"onnx:            {onnx.__version__}")
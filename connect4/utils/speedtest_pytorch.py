import torch
import time

# Vérification du GPU
if not torch.cuda.is_available():
    raise SystemError("CUDA n'est pas disponible.")

print(f"GPU détecté : {torch.cuda.get_device_name(0)}")

# Taille de la matrice (4096x4096)
N = 4096
iterations = 100

# 1. Test FP32
x_32 = torch.randn(N, N, device='cuda', dtype=torch.float32)
torch.cuda.synchronize()
start = time.time()
for _ in range(iterations):
    y_32 = torch.matmul(x_32, x_32)
torch.cuda.synchronize()
t_fp32 = time.time() - start
print(f"Temps FP32 : {t_fp32:.4f} s")

# 2. Test FP16
x_16 = x_32.half()
torch.cuda.synchronize()
start = time.time()
for _ in range(iterations):
    y_16 = torch.matmul(x_16, x_16)
torch.cuda.synchronize()
t_fp16 = time.time() - start
print(f"Temps FP16 : {t_fp16:.4f} s")

# 3. Test INT8 (DP4A)
x_8 = torch.randint(-128, 127, (N, N), device='cuda', dtype=torch.int8)
torch.cuda.synchronize()
start = time.time()
for _ in range(iterations):
    y_8 = torch._int_mm(x_8, x_8)
torch.cuda.synchronize()
t_int8 = time.time() - start
print(f"Temps INT8 : {t_int8:.4f} s")

# Bilan des accélérations par rapport au FP32
print("\n--- Ratios de vitesse (par rapport au FP32) ---")
print(f"Accélération FP16 : {t_fp32 / t_fp16:.2f}x")
print(f"Accélération INT8 : {t_fp32 / t_int8:.2f}x")
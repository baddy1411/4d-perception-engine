"""GPU vs CPU benchmark for LiDAR projection."""
import cupy as cp
import numpy as np
import time

# Benchmark GPU projection
N = 1000000  # 1M points
points = np.random.randn(N, 4).astype(np.float32) * 50
extrinsic = np.eye(4, dtype=np.float32)
intrinsic = np.array([[800,0,400],[0,800,300],[0,0,1]], dtype=np.float32)

print(f"Benchmarking with {N:,} points on RTX 3060 Ti...")
print(f"CUDA device: {cp.cuda.runtime.getDeviceCount()} GPU(s)")

# GPU version
points_gpu = cp.asarray(points[:,:3])
ext_gpu = cp.asarray(extrinsic)

# Warmup
ones = cp.ones((N, 1), dtype=cp.float32)
xyz_h = cp.hstack([points_gpu, ones])
cam = (ext_gpu @ xyz_h.T).T
cp.cuda.Stream.null.synchronize()

# GPU Benchmark
start = time.perf_counter()
for _ in range(10):
    ones = cp.ones((N, 1), dtype=cp.float32)
    xyz_h = cp.hstack([points_gpu, ones])
    cam = (ext_gpu @ xyz_h.T).T
    cp.cuda.Stream.null.synchronize()
gpu_time = (time.perf_counter() - start) / 10

print(f"GPU: {gpu_time*1000:.2f} ms for 1M points")
print(f"GPU: {N / gpu_time / 1e6:.1f} M points/sec")

# CPU version (numpy)
start = time.perf_counter()
for _ in range(10):
    ones = np.ones((N, 1), dtype=np.float32)
    xyz_h = np.hstack([points[:,:3], ones])
    cam = (extrinsic @ xyz_h.T).T
cpu_time = (time.perf_counter() - start) / 10

print(f"CPU: {cpu_time*1000:.2f} ms for 1M points")
print(f"CPU: {N / cpu_time / 1e6:.1f} M points/sec")
print(f"Speedup: {cpu_time / gpu_time:.1f}x")

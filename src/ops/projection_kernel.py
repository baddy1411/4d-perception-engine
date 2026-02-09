"""
LiDAR-to-Camera Projection Kernel
CPU and GPU implementations using NumPy/CuPy.

This provides the same functionality as the CUDA kernel in projection.cu
but works without requiring nvcc compilation.
"""

import numpy as np
from typing import Tuple, Optional

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


def project_lidar_to_camera_cpu(
    points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    img_width: int = 800,
    img_height: int = 600
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project 3D LiDAR points to 2D camera image plane (CPU version).
    
    This implements the same algorithm as the CUDA kernel:
    1. Apply extrinsic (4x4 homogeneous transformation) to transform from world to camera frame
    2. Apply intrinsic (3x3) to project to image coordinates
    3. Perform perspective division
    4. Filter points outside image bounds
    
    Args:
        points: (N, 3) or (N, 4) LiDAR points in world coordinates
        extrinsic: (4, 4) world-to-camera transformation matrix
        intrinsic: (3, 3) camera intrinsic matrix
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        uv: (M, 2) valid pixel coordinates
        depths: (M,) depth values for valid points
        mask: (N,) boolean mask of valid points
    """
    N = len(points)
    
    # Extract xyz (handle both N,3 and N,4 formats)
    xyz = points[:, :3]
    
    # Convert to homogeneous coordinates (N, 4)
    ones = np.ones((N, 1), dtype=xyz.dtype)
    xyz_h = np.hstack([xyz, ones])
    
    # Apply extrinsic transformation: world -> camera
    # cam_coords = extrinsic @ xyz_h.T = (4, N)
    cam_coords = (extrinsic @ xyz_h.T).T  # (N, 4)
    
    # Extract camera-frame coordinates
    x_cam = cam_coords[:, 0]
    y_cam = cam_coords[:, 1]
    z_cam = cam_coords[:, 2]
    
    # Filter points behind camera (z <= 0)
    valid_depth = z_cam > 0.1
    
    # Apply intrinsic projection for valid points only
    valid_indices = np.where(valid_depth)[0]
    x_valid = x_cam[valid_indices]
    y_valid = y_cam[valid_indices]
    z_valid = z_cam[valid_indices]
    
    # Project to image plane: [u, v, 1]^T = K @ [x/z, y/z, 1]^T
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    u = (fx * x_valid / z_valid) + cx
    v = (fy * y_valid / z_valid) + cy
    
    # Filter points outside image bounds
    valid_bounds = (
        (u >= 0) & (u < img_width) &
        (v >= 0) & (v < img_height)
    )
    
    # Create output arrays
    uv = np.stack([u[valid_bounds], v[valid_bounds]], axis=1)
    depths = z_valid[valid_bounds]
    
    # Create full mask
    mask = np.zeros(N, dtype=bool)
    mask[valid_indices[valid_bounds]] = True
    
    return uv, depths, mask


def project_lidar_to_camera_gpu(
    points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    img_width: int = 800,
    img_height: int = 600
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GPU-accelerated projection using CuPy.
    
    Same algorithm as CPU version but runs on NVIDIA GPU.
    Falls back to CPU if CuPy is not available.
    """
    if not CUPY_AVAILABLE:
        print("CuPy not available, falling back to CPU")
        return project_lidar_to_camera_cpu(
            points, extrinsic, intrinsic, img_width, img_height
        )
    
    N = len(points)
    
    # Transfer to GPU
    points_gpu = cp.asarray(points[:, :3])
    extrinsic_gpu = cp.asarray(extrinsic)
    
    # Add homogeneous coordinate
    ones = cp.ones((N, 1), dtype=points_gpu.dtype)
    xyz_h = cp.hstack([points_gpu, ones])
    
    # Apply extrinsic transformation
    cam_coords = (extrinsic_gpu @ xyz_h.T).T
    
    x_cam = cam_coords[:, 0]
    y_cam = cam_coords[:, 1]
    z_cam = cam_coords[:, 2]
    
    # Filter points behind camera
    valid_depth = z_cam > 0.1
    valid_indices = cp.where(valid_depth)[0]
    
    x_valid = x_cam[valid_indices]
    y_valid = y_cam[valid_indices]
    z_valid = z_cam[valid_indices]
    
    # Project to image plane
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    
    u = (fx * x_valid / z_valid) + cx
    v = (fy * y_valid / z_valid) + cy
    
    # Filter points outside image bounds
    valid_bounds = (
        (u >= 0) & (u < img_width) &
        (v >= 0) & (v < img_height)
    )
    
    # Transfer back to CPU
    uv = cp.asnumpy(cp.stack([u[valid_bounds], v[valid_bounds]], axis=1))
    depths = cp.asnumpy(z_valid[valid_bounds])
    
    mask = np.zeros(N, dtype=bool)
    valid_cpu = cp.asnumpy(valid_indices[valid_bounds])
    mask[valid_cpu] = True
    
    return uv, depths, mask


def project_points(
    points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    img_width: int = 800,
    img_height: int = 600,
    use_gpu: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project LiDAR points to camera image.
    
    Automatically selects GPU or CPU based on availability.
    
    Args:
        points: (N, 3) or (N, 4) point cloud
        extrinsic: (4, 4) world-to-camera transform
        intrinsic: (3, 3) camera intrinsics
        img_width: Image width
        img_height: Image height
        use_gpu: Whether to try GPU acceleration
        
    Returns:
        uv: (M, 2) pixel coordinates
        depths: (M,) depth values
        mask: (N,) boolean validity mask
    """
    if use_gpu and CUPY_AVAILABLE:
        return project_lidar_to_camera_gpu(
            points, extrinsic, intrinsic, img_width, img_height
        )
    else:
        return project_lidar_to_camera_cpu(
            points, extrinsic, intrinsic, img_width, img_height
        )


def benchmark_projection(num_points: int = 1000000, num_runs: int = 10):
    """Benchmark CPU vs GPU projection performance."""
    import time
    
    # Generate random point cloud
    points = np.random.randn(num_points, 4).astype(np.float32) * 50
    
    # Define camera parameters
    extrinsic = np.eye(4, dtype=np.float32)
    intrinsic = np.array([
        [800, 0, 400],
        [0, 800, 300],
        [0, 0, 1]
    ], dtype=np.float32)
    
    print(f"Benchmarking projection kernel with {num_points:,} points...")
    
    # Warmup
    project_lidar_to_camera_cpu(points, extrinsic, intrinsic)
    
    # CPU benchmark
    start = time.perf_counter()
    for _ in range(num_runs):
        uv_cpu, depths_cpu, mask_cpu = project_lidar_to_camera_cpu(
            points, extrinsic, intrinsic
        )
    cpu_time = (time.perf_counter() - start) / num_runs
    
    print(f"  CPU: {cpu_time*1000:.2f} ms/iteration")
    print(f"  CPU: {num_points / cpu_time / 1e6:.2f} M points/sec")
    print(f"  Valid points: {mask_cpu.sum():,}")
    
    # GPU benchmark (if available)
    if CUPY_AVAILABLE:
        # Warmup
        project_lidar_to_camera_gpu(points, extrinsic, intrinsic)
        cp.cuda.Stream.null.synchronize()
        
        start = time.perf_counter()
        for _ in range(num_runs):
            uv_gpu, depths_gpu, mask_gpu = project_lidar_to_camera_gpu(
                points, extrinsic, intrinsic
            )
            cp.cuda.Stream.null.synchronize()
        gpu_time = (time.perf_counter() - start) / num_runs
        
        print(f"  GPU: {gpu_time*1000:.2f} ms/iteration")
        print(f"  GPU: {num_points / gpu_time / 1e6:.2f} M points/sec")
        print(f"  Speedup: {cpu_time / gpu_time:.1f}x")
    else:
        print("  GPU: CuPy not available")


if __name__ == "__main__":
    # Test projection
    print("Testing LiDAR projection kernel...")
    
    # Create test point cloud
    points = np.random.randn(10000, 4).astype(np.float32) * 20
    
    extrinsic = np.eye(4, dtype=np.float32)
    intrinsic = np.array([
        [400, 0, 400],
        [0, 400, 300],
        [0, 0, 1]
    ], dtype=np.float32)
    
    uv, depths, mask = project_points(points, extrinsic, intrinsic)
    
    print(f"Input: {len(points)} points")
    print(f"Output: {len(uv)} projected points")
    print(f"Valid: {mask.sum()} points in camera FOV")
    
    print("\n" + "="*50)
    benchmark_projection(num_points=100000, num_runs=5)

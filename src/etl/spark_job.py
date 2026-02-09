"""
Distributed ETL Pipeline for OPV2V Perception Data - GPU Accelerated
PySpark job with custom spatial partitioners for point cloud processing.

This demonstrates:
1. GPU-accelerated point cloud generation using CuPy
2. Custom spatial partitioning
3. Parallel processing of depth images across scenes
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, List, Dict
from dataclasses import dataclass
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from pyspark.sql import SparkSession
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False

from PIL import Image

# GPU Acceleration
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    print("CuPy not available. Running on CPU.")


class SpatialPartitioner:
    """
    Custom spatial partitioner for point cloud data.
    """
    
    def __init__(
        self, 
        num_partitions: int = 64,
        x_range: Tuple[float, float] = (-100, 100),
        y_range: Tuple[float, float] = (-100, 100),
        z_range: Tuple[float, float] = (-10, 50)
    ):
        self.num_partitions = num_partitions
        
        # Calculate grid dimensions 
        self.grid_x = int(np.cbrt(num_partitions)) + 1
        self.grid_y = int(np.cbrt(num_partitions)) + 1
        self.grid_z = max(1, num_partitions // (self.grid_x * self.grid_y))
        
        # Cell sizes
        self.cell_x = (x_range[1] - x_range[0]) / self.grid_x
        self.cell_y = (y_range[1] - y_range[0]) / self.grid_y
        self.cell_z = (z_range[1] - z_range[0]) / self.grid_z
        
        self.x_min, self.y_min, self.z_min = x_range[0], y_range[0], z_range[0]
        
    def get_partition(self, x, y, z):
        """Get partition ID for a 3D point."""
        ix = ((x - self.x_min) / self.cell_x).astype(int)
        iy = ((y - self.y_min) / self.cell_y).astype(int)
        iz = ((z - self.z_min) / self.cell_z).astype(int)
        
        # Clamp
        ix = np.clip(ix, 0, self.grid_x - 1)
        iy = np.clip(iy, 0, self.grid_y - 1)
        iz = np.clip(iz, 0, self.grid_z - 1)
        
        return (ix * self.grid_y * self.grid_z + iy * self.grid_z + iz) % self.num_partitions
    
    def partition_points_gpu(self, points_gpu):
        """Partition point cloud using GPU."""
        if not HAS_GPU:
            return self.partition_points_cpu(points_gpu)
            
        x, y, z = points_gpu[:, 0], points_gpu[:, 1], points_gpu[:, 2]
        
        pids = self.get_partition(x, y, z)
        
        # Group by partition (this part is tricky on GPU, moving to CPU for grouping)
        pids_cpu = cp.asnumpy(pids)
        points_cpu = cp.asnumpy(points_gpu)
        
        partitions = {}
        # Simple loop for demo (can be optimized with sort)
        unique_pids = np.unique(pids_cpu)
        for pid in unique_pids:
            mask = (pids_cpu == pid)
            partitions[int(pid)] = points_cpu[mask]
            
        return partitions

    def partition_points_cpu(self, points):
        """Partition point cloud using CPU."""
        partitions = {}
        for i, point in enumerate(points):
            ix = int((point[0] - self.x_min) / self.cell_x)
            iy = int((point[1] - self.y_min) / self.cell_y)
            iz = int((point[2] - self.z_min) / self.cell_z)
            
            ix = max(0, min(ix, self.grid_x - 1))
            iy = max(0, min(iy, self.grid_y - 1))
            iz = max(0, min(iz, self.grid_z - 1))
            
            pid = (ix * self.grid_y * self.grid_z + iy * self.grid_z + iz) % self.num_partitions
            
            if pid not in partitions:
                partitions[pid] = []
            partitions[pid].append(i)
        
        return {k: points[np.array(v)] for k, v in partitions.items()}


def load_depth_image(filepath: str) -> np.ndarray:
    """Load a depth image and convert to meters."""
    img = Image.open(filepath)
    depth = np.array(img, dtype=np.float32)
    
    # Normalize to depth in meters
    if depth.max() > 255:
        depth = depth / 65535.0 * 100.0
    else:
        depth = depth / 255.0 * 100.0
    
    return depth


def depth_to_points_gpu(
    depth: np.ndarray,
    fx: float = 400.0,
    fy: float = 400.0,
    cx: float = 400.0,
    cy: float = 300.0
):
    """Convert depth image to 3D point cloud using GPU."""
    if not HAS_GPU:
        return depth_to_points_cpu(depth, fx, fy, cx, cy)
    
    H, W = depth.shape
    depth_gpu = cp.asarray(depth)
    
    # Create grid on GPU
    u_gpu, v_gpu = cp.meshgrid(cp.arange(W), cp.arange(H))
    
    # projection
    z = depth_gpu.flatten()
    u = u_gpu.flatten()
    v = v_gpu.flatten()
    
    x = ((u - cx) * z) / fx
    y = ((v - cy) * z) / fy
    
    # Filter invalid points
    valid_mask = (z > 0.1) & (z < 100)
    
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    
    points = cp.vstack([x, y, z]).T
    return points


def depth_to_points_cpu(depth, fx=400.0, fy=400.0, cx=400.0, cy=300.0):
    H, W = depth.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    z = depth.flatten()
    x = ((u.flatten() - cx) * z) / fx
    y = ((v.flatten() - cy) * z) / fy
    
    points = np.vstack([x, y, z]).T
    valid = (z > 0.1) & (z < 100)
    return points[valid]


def process_frame(
    frame_path: str,
    partitioner: SpatialPartitioner
) -> Dict:
    """Process a single frame using GPU if available."""
    
    # Load depth image (disk I/O on CPU)
    depth = load_depth_image(frame_path)
    
    # GPU Processing
    if HAS_GPU:
        points = depth_to_points_gpu(depth)
        # Partitioning
        partitioned = partitioner.partition_points_gpu(points)
        num_points = len(points)
        min_depth = float(cp.min(points[:, 2]))
        max_depth = float(cp.max(points[:, 2]))
    else:
        points = depth_to_points_cpu(depth)
        partitioned = partitioner.partition_points_cpu(points)
        num_points = len(points)
        min_depth = float(np.min(points[:, 2]))
        max_depth = float(np.max(points[:, 2]))
    
    return {
        'frame_path': frame_path,
        'total_points': num_points,
        'partition_counts': {k: len(v) for k, v in partitioned.items()},
        'num_partitions_used': len(partitioned),
        'depth_range': (min_depth, max_depth)
    }


def run_pipeline(data_dir: Path, max_frames: int = 1000):
    """Run pipeline."""
    
    print(f"Running {'GPU' if HAS_GPU else 'CPU'} Accelerated Pipeline")
    print("="*60)
    
    if HAS_GPU:
        print(f"GPU Device: {cp.cuda.runtime.getDeviceCount()} devices")
        # Warmup
        depth_to_points_gpu(np.zeros((600, 800), dtype=np.float32))
    
    partitioner = SpatialPartitioner(num_partitions=64)
    
    # Find all depth images
    # Using glob recursive
    hetero_dir = data_dir / "OPV2V_Hetero"
    depth_files = list(hetero_dir.glob("**/*_depth0.png"))
    
    if max_frames:
        depth_files = depth_files[:max_frames]
    
    print(f"Processing {len(depth_files)} frames...")
    
    results = []
    total_points = 0
    start_time = time.time()
    
    for i, frame_path in enumerate(depth_files):
        result = process_frame(str(frame_path), partitioner)
        results.append(result)
        total_points += result['total_points']
        
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start_time
            fps = (i + 1) / elapsed
            print(f"  Processed {i+1}/{len(depth_files)} frames | {fps:.1f} FPS | "
                  f"{total_points:,} points total")
    
    total_time = time.time() - start_time
    print(f"\nPipeline complete in {total_time:.2f}s")
    print(f"  Average FPS: {len(results)/total_time:.1f}")
    print(f"  Total points: {total_points:,}")
    
    return results


if __name__ == "__main__":
    # Handle paths for WSL/Windows
    if os.path.exists("/mnt/c/Users/badri/Desktop/4D/data/opv2v"):
         data_dir = Path("/mnt/c/Users/badri/Desktop/4D/data/opv2v")
    elif os.path.exists("C:/Users/badri/Desktop/4D/data/opv2v"):
         data_dir = Path("C:/Users/badri/Desktop/4D/data/opv2v")
    else:
         # Fallback to current dir relative
         data_dir = Path("data/opv2v")
         
    # Run full dataset if no arg provided, or limit if testing
    # User asked for full dataset
    results = run_pipeline(data_dir, max_frames=None)
    
    output_file = data_dir / "etl_results.json"
    print(f"Saving results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print("Done!")

"""
Data Loader for OPV2V and Synthetic Point Cloud Data
Loads LiDAR point clouds and camera images for the 4D Perception Engine.
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator

class PointCloudLoader:
    """Loads and processes LiDAR point cloud data."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.scenes = self._discover_scenes()
        
    def _discover_scenes(self) -> List[Path]:
        """Discover all available scenes in the data directory."""
        scenes = []
        
        # Check for sample data structure
        sample_dir = self.data_dir / "sample"
        if sample_dir.exists():
            scenes.extend(sorted(sample_dir.glob("scene_*")))
        
        # Check for OPV2V structure (train/validate/test splits)
        for split in ['train', 'validate', 'test']:
            split_dir = self.data_dir / split
            if split_dir.exists():
                scenes.extend(sorted(split_dir.glob("*/")))
        
        return scenes
    
    def load_point_cloud(self, filepath: Path) -> np.ndarray:
        """
        Load a point cloud from binary file.
        
        Args:
            filepath: Path to .bin or .pcd file
            
        Returns:
            points: (N, 4) array of [x, y, z, intensity]
        """
        if filepath.suffix == '.bin':
            # KITTI-style binary format
            points = np.fromfile(str(filepath), dtype=np.float32)
            points = points.reshape(-1, 4)
        elif filepath.suffix == '.pcd':
            # PCD format (simplified parsing)
            points = self._load_pcd(filepath)
        else:
            raise ValueError(f"Unsupported format: {filepath.suffix}")
        
        return points
    
    def _load_pcd(self, filepath: Path) -> np.ndarray:
        """Load point cloud from PCD file (simplified)."""
        with open(filepath, 'rb') as f:
            # Skip header
            while True:
                line = f.readline().decode('utf-8').strip()
                if line.startswith('DATA'):
                    break
            
            # Read binary data
            points = np.frombuffer(f.read(), dtype=np.float32)
            
        # Assume 4 fields (x, y, z, intensity)
        try:
            points = points.reshape(-1, 4)
        except ValueError:
            # If can't reshape to 4, try 3 (xyz only) and add zeros
            points = points.reshape(-1, 3)
            points = np.hstack([points, np.zeros((len(points), 1))])
        
        return points
    
    def iter_frames(self, scene_idx: int = 0) -> Generator[Dict, None, None]:
        """
        Iterate over all frames in a scene.
        
        Yields:
            dict with keys: 'points', 'frame_id', 'scene_id', 'vehicle_id'
        """
        if scene_idx >= len(self.scenes):
            raise IndexError(f"Scene {scene_idx} not found. Available: {len(self.scenes)}")
        
        scene_dir = self.scenes[scene_idx]
        
        # Find all vehicle directories
        vehicle_dirs = sorted(scene_dir.glob("vehicle_*"))
        if not vehicle_dirs:
            vehicle_dirs = [scene_dir]  # Single vehicle scene
        
        for vehicle_dir in vehicle_dirs:
            lidar_dir = vehicle_dir / "velodyne"
            if not lidar_dir.exists():
                continue
            
            for lidar_file in sorted(lidar_dir.glob("*.bin")):
                points = self.load_point_cloud(lidar_file)
                
                yield {
                    'points': points,
                    'frame_id': lidar_file.stem,
                    'scene_id': scene_dir.name,
                    'vehicle_id': vehicle_dir.name,
                    'filepath': str(lidar_file)
                }
    
    def get_calibration(self, scene_idx: int = 0) -> Dict:
        """Load calibration data for sensor fusion."""
        scene_dir = self.scenes[scene_idx] if scene_idx < len(self.scenes) else self.data_dir
        
        calib_file = scene_dir / "calib.json"
        if not calib_file.exists():
            # Check parent directories
            calib_file = self.data_dir / "sample" / "calib.json"
        
        if calib_file.exists():
            with open(calib_file) as f:
                return json.load(f)
        
        # Return default calibration
        return {
            "extrinsic": np.eye(4).tolist(),
            "intrinsic": [[800, 0, 400], [0, 800, 300], [0, 0, 1]]
        }
    
    def __len__(self) -> int:
        return len(self.scenes)
    
    def __repr__(self) -> str:
        return f"PointCloudLoader(scenes={len(self.scenes)}, data_dir='{self.data_dir}')"


def project_points_to_image(
    points: np.ndarray,
    extrinsic: np.ndarray,
    intrinsic: np.ndarray,
    img_width: int = 800,
    img_height: int = 600
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D LiDAR points to 2D camera image plane.
    
    This is the CPU reference implementation of the CUDA kernel.
    
    Args:
        points: (N, 3) or (N, 4) point cloud
        extrinsic: (4, 4) world-to-camera transformation
        intrinsic: (3, 3) camera intrinsic matrix
        img_width: Image width in pixels
        img_height: Image height in pixels
        
    Returns:
        uv: (M, 2) valid pixel coordinates
        mask: (N,) boolean mask of valid points
    """
    # Extract xyz
    xyz = points[:, :3]
    
    # Convert to homogeneous coordinates
    ones = np.ones((xyz.shape[0], 1))
    xyz_h = np.hstack([xyz, ones])  # (N, 4)
    
    # Apply extrinsic (world -> camera)
    cam_coords = (extrinsic @ xyz_h.T).T  # (N, 4)
    
    # Filter points behind camera
    valid_depth = cam_coords[:, 2] > 0.1
    
    # Apply intrinsic (camera -> image)
    cam_xyz = cam_coords[:, :3]
    cam_xyz = cam_xyz[valid_depth]
    
    # Perspective division
    uv_h = (intrinsic @ cam_xyz.T).T  # (M, 3)
    uv = uv_h[:, :2] / uv_h[:, 2:3]  # (M, 2)
    
    # Filter points outside image bounds
    valid_bounds = (
        (uv[:, 0] >= 0) & (uv[:, 0] < img_width) &
        (uv[:, 1] >= 0) & (uv[:, 1] < img_height)
    )
    
    # Create full mask
    full_mask = np.zeros(len(points), dtype=bool)
    full_mask[valid_depth] = valid_bounds
    
    return uv[valid_bounds], full_mask


# Test the loader
if __name__ == "__main__":
    import sys
    
    # Get project root (4D folder)
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "opv2v"
    
    loader = PointCloudLoader(data_dir)
    print(f"Loader: {loader}")
    
    if len(loader) == 0:
        print("No data found. Run scripts/download_data.py first.")
        sys.exit(1)
    
    # Load first frame
    for frame in loader.iter_frames(0):
        print(f"\nFrame: {frame['frame_id']}")
        print(f"  Scene: {frame['scene_id']}")
        print(f"  Vehicle: {frame['vehicle_id']}")
        print(f"  Points: {frame['points'].shape}")
        
        # Test projection
        calib = loader.get_calibration(0)
        extrinsic = np.array(calib['extrinsic'])
        intrinsic = np.array(calib['intrinsic'])
        
        uv, mask = project_points_to_image(
            frame['points'],
            extrinsic,
            intrinsic
        )
        print(f"  Projected: {len(uv)} points visible in camera")
        
        break  # Just test first frame

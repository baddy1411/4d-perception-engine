"""
OPV2V Dataset Downloader
Downloads a ~4-5GB subset of the OPV2V dataset for the 4D Perception Engine demo.

The OPV2V dataset is hosted on UCLA Box. This script provides instructions
and utilities for downloading a proportional subset.
"""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# Dataset info
DATASET_INFO = """
=============================================================
OPV2V Dataset Download Instructions
=============================================================

The OPV2V dataset is hosted on UCLA Box and requires manual download.

STEP 1: Go to the UCLA Box link:
    https://ucla.app.box.com/v/UCLA-MobilityLab-OPV2V

STEP 2: Download these files (~4-5GB total):
    - validate.zip (validation split - recommended for demo)
    - OR download specific scene folders for a smaller subset

STEP 3: Extract to:
    {data_dir}

STEP 4: Run this script again to verify the data structure.

=============================================================
"""

def check_data_structure(data_dir: Path) -> bool:
    """Check if OPV2V data is properly structured."""
    expected_dirs = ['validate', 'train', 'test']
    
    for split in expected_dirs:
        split_dir = data_dir / split
        if split_dir.exists():
            # Check for scene folders
            scenes = list(split_dir.glob('*/'))
            if scenes:
                # Check for expected files in first scene
                first_scene = scenes[0]
                lidar_files = list(first_scene.glob('**/velodyne/*.pcd'))
                camera_files = list(first_scene.glob('**/camera*/*.png'))
                
                print(f"✓ Found {split}: {len(scenes)} scenes")
                if lidar_files:
                    print(f"  - LiDAR files: {len(lidar_files)} in first scene")
                if camera_files:
                    print(f"  - Camera files: {len(camera_files)} in first scene")
                return True
    
    return False

def create_sample_structure(data_dir: Path):
    """Create sample directory structure for testing without full dataset."""
    import numpy as np
    
    print("Creating synthetic sample data for testing...")
    
    sample_dir = data_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    for scene_idx in range(3):
        scene_dir = sample_dir / f"scene_{scene_idx:04d}"
        
        for vehicle_idx in range(2):
            vehicle_dir = scene_dir / f"vehicle_{vehicle_idx}"
            
            # Create LiDAR directory
            lidar_dir = vehicle_dir / "velodyne"
            lidar_dir.mkdir(parents=True, exist_ok=True)
            
            # Create camera directory
            camera_dir = vehicle_dir / "camera0"
            camera_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate synthetic LiDAR frames
            for frame_idx in range(10):
                # Random point cloud (N x 4: x, y, z, intensity)
                num_points = np.random.randint(50000, 100000)
                points = np.random.randn(num_points, 4).astype(np.float32)
                points[:, :3] *= 50  # Scale to realistic range
                points[:, 3] = np.random.uniform(0, 1, num_points)  # Intensity
                
                # Save as binary (similar to KITTI format)
                lidar_path = lidar_dir / f"{frame_idx:06d}.bin"
                points.tofile(str(lidar_path))
                
            print(f"  Created scene_{scene_idx:04d}/vehicle_{vehicle_idx}")
    
    # Create calibration file
    calib_data = {
        "extrinsic": np.eye(4).tolist(),
        "intrinsic": [[800, 0, 400], [0, 800, 300], [0, 0, 1]]
    }
    
    import json
    with open(sample_dir / "calib.json", "w") as f:
        json.dump(calib_data, f, indent=2)
    
    print(f"\n✓ Created synthetic sample data at: {sample_dir}")
    print(f"  - 3 scenes, 2 vehicles each, 10 frames per vehicle")
    print(f"  - Total: 60 LiDAR frames (~300MB synthetic data)")
    
    return sample_dir

def main():
    data_dir = Path(__file__).parent.parent / "data" / "opv2v"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print(DATASET_INFO.format(data_dir=data_dir))
    
    if check_data_structure(data_dir):
        print("\n✓ OPV2V data found and verified!")
        return
    
    print("\n⚠ OPV2V data not found.")
    print("\nWould you like to create synthetic sample data for testing? (y/n)")
    
    # For automated runs, create sample data
    if len(sys.argv) > 1 and sys.argv[1] == "--create-sample":
        create_sample_structure(data_dir)
    else:
        response = input().strip().lower()
        if response == 'y':
            create_sample_structure(data_dir)
        else:
            print("Please download the OPV2V dataset manually.")

if __name__ == "__main__":
    main()

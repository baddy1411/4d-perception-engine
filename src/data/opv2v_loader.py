"""
Data Loader for OPV2V-H Depth Camera Data
Loads depth images and BEV visibility maps from the OPV2V-H dataset.
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Generator
from PIL import Image

class OPV2VLoader:
    """Loads and processes OPV2V-H depth camera data."""
    
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.hetero_dir = self.data_dir / "OPV2V_Hetero"
        self.scenes = self._discover_scenes()
        
    def _discover_scenes(self) -> List[Dict]:
        """Discover all available scenes in the data directory."""
        scenes = []
        
        for split in ['train', 'validate', 'test']:
            split_dir = self.hetero_dir / split
            if not split_dir.exists():
                continue
            
            # Each scene is a timestamp folder
            for scene_dir in sorted(split_dir.iterdir()):
                if not scene_dir.is_dir():
                    continue
                
                # Each scene has vehicle folders
                for vehicle_dir in sorted(scene_dir.iterdir()):
                    if not vehicle_dir.is_dir():
                        continue
                    
                    # Count frames
                    depth_files = list(vehicle_dir.glob("*_depth0.png"))
                    if depth_files:
                        scenes.append({
                            'split': split,
                            'scene_id': scene_dir.name,
                            'vehicle_id': vehicle_dir.name,
                            'path': vehicle_dir,
                            'num_frames': len(depth_files)
                        })
        
        return scenes
    
    def load_depth_image(self, filepath: Path) -> np.ndarray:
        """
        Load a depth image and convert to depth values.
        
        Args:
            filepath: Path to depth PNG file
            
        Returns:
            depth: (H, W) array of depth values in meters
        """
        img = Image.open(filepath)
        depth = np.array(img, dtype=np.float32)
        
        # Convert to depth (typical encoding: depth = pixel_value / 255 * max_depth)
        # OPV2V uses 16-bit depth, normalize accordingly
        if depth.dtype == np.uint16:
            depth = depth / 65535.0 * 100.0  # Max depth ~100m
        else:
            depth = depth / 255.0 * 100.0  # Fallback for 8-bit
        
        return depth
    
    def load_bev_visibility(self, filepath: Path) -> np.ndarray:
        """Load Bird's Eye View visibility map."""
        img = Image.open(filepath)
        return np.array(img)
    
    def iter_frames(self, scene_idx: int = 0, max_frames: int = 50) -> Generator[Dict, None, None]:
        """
        Iterate over all frames in a scene.
        
        Yields:
            dict with keys: 'depth_images', 'bev', 'frame_id', 'scene_info'
        """
        if scene_idx >= len(self.scenes):
            raise IndexError(f"Scene {scene_idx} not found. Available: {len(self.scenes)}")
        
        scene = self.scenes[scene_idx]
        vehicle_dir = scene['path']
        
        # Find unique frame IDs
        frame_ids = set()
        for f in vehicle_dir.glob("*_depth0.png"):
            frame_id = f.stem.replace("_depth0", "")
            frame_ids.add(frame_id)
        
        frame_ids = sorted(frame_ids)[:max_frames]
        
        for frame_id in frame_ids:
            # Load all 4 depth cameras
            depth_images = []
            for cam_idx in range(4):
                depth_file = vehicle_dir / f"{frame_id}_depth{cam_idx}.png"
                if depth_file.exists():
                    depth = self.load_depth_image(depth_file)
                    depth_images.append(depth)
            
            # Load BEV visibility
            bev_file = vehicle_dir / f"{frame_id}_bev_visibility.png"
            bev = None
            if bev_file.exists():
                bev = self.load_bev_visibility(bev_file)
            
            yield {
                'depth_images': depth_images,
                'bev': bev,
                'frame_id': frame_id,
                'scene_info': scene
            }
    
    def depth_to_point_cloud(
        self, 
        depth: np.ndarray, 
        fx: float = 400.0, 
        fy: float = 400.0,
        cx: float = 400.0,
        cy: float = 300.0
    ) -> np.ndarray:
        """
        Convert depth image to 3D point cloud.
        
        Args:
            depth: (H, W) depth image
            fx, fy: Focal lengths
            cx, cy: Principal point
            
        Returns:
            points: (N, 3) point cloud [x, y, z]
        """
        H, W = depth.shape
        
        # Create pixel coordinate grid
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        
        # Convert to 3D
        z = depth.flatten()
        x = ((u.flatten() - cx) * z) / fx
        y = ((v.flatten() - cy) * z) / fy
        
        # Stack and filter invalid points
        points = np.vstack([x, y, z]).T
        valid = (z > 0.1) & (z < 100)  # Filter invalid depths
        points = points[valid]
        
        return points
    
    def __len__(self) -> int:
        return len(self.scenes)
    
    def __repr__(self) -> str:
        return f"OPV2VLoader(scenes={len(self.scenes)}, data_dir='{self.data_dir}')"


# Test the loader
if __name__ == "__main__":
    import sys
    
    # Get project root (4D folder)
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "opv2v"
    
    loader = OPV2VLoader(data_dir)
    print(f"Loader: {loader}")
    
    if len(loader) == 0:
        print("No data found!")
        sys.exit(1)
    
    print(f"\nFound {len(loader)} scene/vehicle combinations:")
    for i, scene in enumerate(loader.scenes[:5]):
        print(f"  [{i}] {scene['split']}/{scene['scene_id']}/{scene['vehicle_id']} - {scene['num_frames']} frames")
    
    # Load first frame
    print("\nLoading first frame...")
    for frame in loader.iter_frames(0):
        print(f"  Frame: {frame['frame_id']}")
        print(f"  Depth cameras: {len(frame['depth_images'])}")
        if frame['depth_images']:
            depth = frame['depth_images'][0]
            print(f"  Depth[0] shape: {depth.shape}")
            print(f"  Depth[0] range: {depth.min():.2f} - {depth.max():.2f} m")
            
            # Convert to point cloud
            points = loader.depth_to_point_cloud(depth)
            print(f"  Point cloud: {len(points)} points")
        
        if frame['bev'] is not None:
            print(f"  BEV shape: {frame['bev'].shape}")
        
        break

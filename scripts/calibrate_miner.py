"""
Calibrate Hard Example Miner Thresholds.
Analyzes a sample of frames to determine varying thresholds.
"""
import numpy as np
from pathlib import Path
import sys
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.opv2v_loader import OPV2VLoader
from src.perception.hard_example_miner import HardExampleMiner

def calibrate():
    data_dir = project_root / "data" / "opv2v"
    loader = OPV2VLoader(data_dir)
    miner = HardExampleMiner()
    
    print(f"Loaded {len(loader)} scenes.")
    
    # Collect metrics from first 50 frames of first 5 scenes
    variances = []
    edges = []
    close_ratios = []
    
    count = 0
    for scene_idx in range(min(5, len(loader))):
        for frame in loader.iter_frames(scene_idx, max_frames=20):
            if not frame['depth_images']:
                continue
                
            depth = frame['depth_images'][0]
            valid_depth = depth[(depth > 0.1) & (depth < 100)]
            
            if len(valid_depth) == 0:
                continue
                
            # Raw metrics
            var = float(np.std(valid_depth))
            edge = miner.compute_edge_density(depth)
            close = miner.compute_close_range_ratio(depth)
            
            variances.append(var)
            edges.append(edge)
            close_ratios.append(close)
            count += 1
            
    print(f"\nAnalyzed {count} frames.")
    print("-" * 40)
    print(f"Metric          | Mean   | Std    | Min    | Max")
    print("-" * 40)
    
    def print_stats(name, data):
        data = np.array(data)
        print(f"{name:<15} | {data.mean():.4f} | {data.std():.4f} | {data.min():.4f} | {data.max():.4f}")
        return data.mean(), data.max()

    mean_var, max_var = print_stats("Depth Variance", variances)
    mean_edge, max_edge = print_stats("Edge Density", edges)
    mean_close, max_close = print_stats("Close Ratio", close_ratios)
    
    print("-" * 40)
    print("Suggested Thresholds (Mean + 1*Std seems reasonable for 'Hard'):")
    print(f"variance_threshold = {np.mean(variances) + np.std(variances):.1f}")
    print(f"edge_threshold = {np.mean(edges) + np.std(edges):.3f}")
    print(f"close_range_threshold = {np.mean(close_ratios) + np.std(close_ratios):.3f}")

if __name__ == "__main__":
    calibrate()

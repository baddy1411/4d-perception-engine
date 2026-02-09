"""
Hard Example Mining for Perception Data
Identifies challenging samples for targeted training.

Hard examples include:
- Scenes with many close-range objects (occlusions)
- High depth variance (complex geometry)
- Sparse point coverage (sensor degradation)
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class HardExampleScore:
    """Scoring for a single frame."""
    frame_id: str
    difficulty_score: float  # 0-1, higher = harder
    depth_variance: float
    edge_density: float
    close_range_ratio: float
    is_hard: bool
    reasons: List[str]


class HardExampleMiner:
    """
    Analyzes frames to identify perceptually challenging samples.
    
    Uses heuristics based on depth statistics to score difficulty:
    - High depth variance = complex scene geometry
    - Many close-range points = potential occlusions
    - High edge density = detailed objects requiring fine-grained detection
    """
    
    def __init__(
        self,
        variance_threshold: float = 20.0,  # Increased from 5.0 (9.68 seen in typical frames)
        close_range_threshold: float = 0.95, # Increased from 0.8 (0.9+ is common)
        edge_threshold: float = 2.0,       # Increased from 1.0 (noise creates high density)
        hard_score_threshold: float = 0.75 # Increased from 0.7
    ):
        self.variance_threshold = variance_threshold
        self.close_range_threshold = close_range_threshold
        self.edge_threshold = edge_threshold
        self.hard_score_threshold = hard_score_threshold
    
    def compute_edge_density(self, depth: np.ndarray) -> float:
        """Compute normalized edge density from Sobel gradients."""
        # Simple gradient magnitude
        dy = np.abs(np.diff(depth, axis=0))
        dx = np.abs(np.diff(depth, axis=1))
        
        # Normalize by image size
        edge_density = (np.mean(dy) + np.mean(dx)) / 2.0
        return float(edge_density)
    
    def compute_close_range_ratio(
        self, 
        depth: np.ndarray, 
        close_threshold: float = 10.0
    ) -> float:
        """Compute ratio of points within close range."""
        valid_depth = depth[(depth > 0.1) & (depth < 100)]
        if len(valid_depth) == 0:
            return 0.0
        
        close_points = np.sum(valid_depth < close_threshold)
        return float(close_points / len(valid_depth))
    
    def score_frame(
        self, 
        depth: np.ndarray, 
        frame_id: str = "unknown"
    ) -> HardExampleScore:
        """
        Score a single frame for perceptual difficulty.
        
        Args:
            depth: (H, W) depth image in meters
            frame_id: Identifier for the frame
            
        Returns:
            HardExampleScore with difficulty metrics
        """
        valid_depth = depth[(depth > 0.1) & (depth < 100)]
        
        if len(valid_depth) == 0:
            return HardExampleScore(
                frame_id=frame_id,
                difficulty_score=0.0,
                depth_variance=0.0,
                edge_density=0.0,
                close_range_ratio=0.0,
                is_hard=False,
                reasons=["Empty depth map"]
            )
        
        # Compute metrics
        depth_variance = float(np.std(valid_depth))
        edge_density = self.compute_edge_density(depth)
        close_range_ratio = self.compute_close_range_ratio(depth)
        
        # Normalize metrics to [0, 1]
        norm_variance = min(depth_variance / self.variance_threshold, 1.0)
        norm_edges = min(edge_density / self.edge_threshold, 1.0)
        norm_close = min(close_range_ratio / self.close_range_threshold, 1.0)
        
        # Weighted difficulty score
        difficulty_score = (
            0.4 * norm_variance +
            0.3 * norm_edges +
            0.3 * norm_close
        )
        
        # Determine if hard example
        reasons = []
        if norm_variance > 0.7:
            reasons.append("High depth variance (complex geometry)")
        if norm_edges > 0.7:
            reasons.append("High edge density (detailed objects)")
        if norm_close > 0.7:
            reasons.append("Many close-range points (occlusion risk)")
        
        is_hard = difficulty_score > self.hard_score_threshold
        
        return HardExampleScore(
            frame_id=frame_id,
            difficulty_score=difficulty_score,
            depth_variance=depth_variance,
            edge_density=edge_density,
            close_range_ratio=close_range_ratio,
            is_hard=is_hard,
            reasons=reasons if reasons else ["Normal scene"]
        )
    
    def score_batch(
        self, 
        depth_images: List[Tuple[str, np.ndarray]]
    ) -> List[HardExampleScore]:
        """Score a batch of frames and sort by difficulty."""
        scores = [
            self.score_frame(depth, frame_id)
            for frame_id, depth in depth_images
        ]
        
        # Sort by difficulty (hardest first)
        scores.sort(key=lambda x: x.difficulty_score, reverse=True)
        return scores
    
    def select_hard_examples(
        self,
        scores: List[HardExampleScore],
        top_k: int = 10
    ) -> List[HardExampleScore]:
        """Select top-k hardest examples for targeted training."""
        return [s for s in scores if s.is_hard][:top_k]


if TORCH_AVAILABLE:
    class NeuralHardExampleClassifier(nn.Module):
        """
        Neural network for learning to predict hard examples.
        
        Takes depth statistics as input and predicts difficulty score.
        Can be trained on labeled hard/easy examples.
        """
        
        def __init__(self, input_dim: int = 8, hidden_dim: int = 32):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)
        
        @staticmethod
        def extract_features(depth: np.ndarray) -> np.ndarray:
            """Extract feature vector from depth image."""
            valid = depth[(depth > 0.1) & (depth < 100)]
            
            if len(valid) == 0:
                return np.zeros(8, dtype=np.float32)
            
            features = np.array([
                np.mean(valid),
                np.std(valid),
                np.min(valid),
                np.max(valid),
                np.median(valid),
                np.percentile(valid, 25),
                np.percentile(valid, 75),
                len(valid) / depth.size  # Coverage ratio
            ], dtype=np.float32)
            
            return features


def run_mining_demo(data_dir: Path) -> None:
    """Run hard example mining on OPV2V data."""
    import sys
    sys.path.insert(0, str(data_dir.parent.parent))
    
    from src.data.opv2v_loader import OPV2VLoader
    
    print("="*60)
    print("Hard Example Mining Demo")
    print("="*60)
    
    loader = OPV2VLoader(data_dir)
    miner = HardExampleMiner()
    
    print(f"Loaded {len(loader)} scene/vehicle combinations")
    
    # Process first scene
    depth_images = []
    for frame in loader.iter_frames(0, max_frames=30):
        depth_images.append((frame['frame_id'], frame['depth_images'][0]))
    
    print(f"Processing {len(depth_images)} frames...")
    
    # Score all frames
    scores = miner.score_batch(depth_images)
    
    # Print results
    print("\nTop 10 Hardest Examples:")
    print("-" * 60)
    for i, score in enumerate(scores[:10]):
        print(f"  {i+1}. Frame {score.frame_id}: "
              f"score={score.difficulty_score:.3f}, "
              f"hard={score.is_hard}")
        print(f"      Reasons: {', '.join(score.reasons)}")
    
    # Summary statistics
    hard_count = sum(1 for s in scores if s.is_hard)
    avg_score = np.mean([s.difficulty_score for s in scores])
    
    print("\nSummary:")
    print(f"  Total frames: {len(scores)}")
    print(f"  Hard examples: {hard_count} ({100*hard_count/len(scores):.1f}%)")
    print(f"  Average difficulty: {avg_score:.3f}")
    
    return scores


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / "opv2v"
    
    if not (data_dir / "OPV2V_Hetero").exists():
        print(f"Error: OPV2V data not found at {data_dir}")
        exit(1)
    
    scores = run_mining_demo(data_dir)

import numpy as np

def filter_ground_plane(points: np.ndarray, threshold: float = -1.5) -> np.ndarray:
    """
    Removes points below a certain z-threshold, assuming z-axis is vertical.
    
    Args:
        points: (N, 3) numpy array of x, y, z coordinates.
        threshold: Z-value below which points are considered ground.
        
    Returns:
        Filtered (M, 3) numpy array.
    """
    return points[points[:, 2] > threshold]

def crop_roi(points: np.ndarray, x_range: tuple, y_range: tuple) -> np.ndarray:
    """
    Crops the point cloud to a region of interest.
    
    Args:
        points: (N, 3) numpy array.
        x_range: (min_x, max_x) tuple.
        y_range: (min_y, max_y) tuple.
        
    Returns:
        Filtered (M, 3) numpy array.
    """
    mask = (
        (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
        (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1])
    )
    return points[mask]

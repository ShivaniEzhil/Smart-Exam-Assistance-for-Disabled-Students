"""
Shared landmark normalization for gesture recognition.
Use the same normalization when training and at inference so the model
sees consistent input (hand shape) regardless of position/size in frame.
"""
import numpy as np

# MediaPipe hand: 21 landmarks, index 0 = WRIST
EXPECTED_SIZE = 21 * 3  # 63


def normalize_landmarks(keypoints):
    """
    Center hand by wrist and scale by hand size so distance from camera doesn't matter.
    Input: (63,) or (21, 3). Output: (63,) float64.
    """
    pts = np.asarray(keypoints, dtype=np.float64)
    if pts.size != EXPECTED_SIZE:
        return pts.flatten() if pts.ndim > 1 else pts
    if pts.ndim == 1:
        pts = pts.reshape(21, 3)
    wrist = pts[0:1]
    pts_centered = pts - wrist
    scale = np.linalg.norm(pts_centered, axis=1).max()
    if scale < 1e-8:
        return keypoints.flatten() if np.asarray(keypoints).ndim > 1 else np.asarray(keypoints, dtype=np.float64)
    pts_norm = pts_centered / scale
    return pts_norm.flatten()

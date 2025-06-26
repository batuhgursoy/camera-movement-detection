import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque

MIN_MATCH_COUNT = 15  # Increased for more robust homography
FLANN_INDEX_LSH = 6

def analyze_homography(M: np.ndarray) -> Dict[str, float]:
    """
    Decomposes a homography matrix to extract directional motion components.
    """
    # Translation components
    dx = M[0, 2]
    dy = M[1, 2]

    # Rotation
    rotation_angle = np.degrees(np.arctan2(M[1, 0], M[0, 0]))

    # Scaling (Zoom)
    scale = (np.sqrt(M[0, 0]**2 + M[1, 0]**2) + np.sqrt(M[0, 1]**2 + M[1, 1]**2)) / 2.0

    return {
        "dx": dx,
        "dy": dy,
        "rotation": rotation_angle,
        "scale": scale
    }


def find_homography_matrix(
    img1: np.ndarray, img2: np.ndarray
) -> Optional[np.ndarray]:
    """
    Computes the homography matrix between two images using ORB and a robust Brute-Force matcher.
    """
    orb = cv2.ORB_create(nfeatures=2000)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    if des1 is None or des2 is None or len(des1) < MIN_MATCH_COUNT or len(des2) < MIN_MATCH_COUNT:
        return None

    # Replaced the brittle FLANN matcher with a robust Brute-Force matcher.
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = [
        m for m, n in matches if len((m,n)) == 2 and m.distance < 0.75 * n.distance
    ]

    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        return M
    return None

def classify_movement(
    dx: float,
    dy: float,
    rotation: float,
    scale: float,
    trans_thresh: float,
    rot_thresh: float,
    scale_thresh_ratio: float,
) -> Optional[str]:
    """Classifies movement with detailed directional information."""
    
    translation_magnitude = np.sqrt(dx**2 + dy**2)
    
    is_zooming_in = scale > 1 + scale_thresh_ratio
    is_zooming_out = scale < 1 - scale_thresh_ratio
    is_rotating_cw = rotation > rot_thresh
    is_rotating_ccw = rotation < -rot_thresh
    is_panning = abs(dx) > abs(dy) and translation_magnitude > trans_thresh
    is_tilting = abs(dy) > abs(dx) and translation_magnitude > trans_thresh

    types = []
    if is_panning:
        types.append("Pan " + ("Right" if dx > 0 else "Left"))
    elif is_tilting:
        types.append("Tilt " + ("Down" if dy > 0 else "Up"))

    if is_rotating_cw:
        types.append("Clockwise Rotation")
    elif is_rotating_ccw:
        types.append("Counter-Clockwise Rotation")
        
    if is_zooming_in:
        types.append("Zoom In")
    elif is_zooming_out:
        types.append("Zoom Out")

    if not types:
        return None
    
    return " & ".join(types)

def detect_significant_movement(
    frames: List[np.ndarray],
    translation_threshold: float = 0.6,
    rotation_threshold: float = 0.3,
    scale_threshold: float = 0.01,
    smoothing_window: int = 3,
) -> List[Dict[str, Any]]:
    """
    Detects significant camera movement using a temporally smoothed approach.
    """
    if len(frames) < 2:
        return []

    movement_indices = []
    motion_metrics_buffer = deque(maxlen=smoothing_window)
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    for idx in range(1, len(frames)):
        current_gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY)
        M = find_homography_matrix(prev_gray, current_gray)

        current_metrics = {"dx": 0, "dy": 0, "rotation": 0, "scale": 1.0}
        if M is not None:
            current_metrics = analyze_homography(M)
        
        motion_metrics_buffer.append(current_metrics)

        if len(motion_metrics_buffer) == smoothing_window:
            avg_dx = np.mean([m["dx"] for m in motion_metrics_buffer])
            avg_dy = np.mean([m["dy"] for m in motion_metrics_buffer])
            avg_rotation = np.mean([m["rotation"] for m in motion_metrics_buffer])
            avg_scale = np.mean([m["scale"] for m in motion_metrics_buffer])

            movement_type = classify_movement(
                avg_dx,
                avg_dy,
                avg_rotation,
                avg_scale,
                translation_threshold,
                rotation_threshold,
                scale_threshold,
            )

            target_frame_idx = idx - smoothing_window // 2
            if movement_type:
                if not movement_indices or movement_indices[-1]["frame_index"] != target_frame_idx:
                    movement_indices.append({
                        "frame_index": target_frame_idx,
                        "type": movement_type,
                        "translation": round(np.sqrt(avg_dx**2 + avg_dy**2), 2),
                        "rotation": round(avg_rotation, 2),
                        "scale": round(avg_scale, 2),
                    })
        
        prev_gray = current_gray

    return movement_indices

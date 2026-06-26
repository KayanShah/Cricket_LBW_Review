import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class BallDetection:
    frame: int
    x: float
    y: float
    confidence: float


def detect_ball_in_frame(frame: np.ndarray, prev_pos: Optional[tuple] = None) -> Optional[tuple]:
    """
    Detect cricket ball using color segmentation + contour analysis.
    Returns (x, y) center or None.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red ball masks (red wraps around in HSV)
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 80])
    upper_red2 = np.array([180, 255, 255])

    # White ball mask
    lower_white = np.array([0, 0, 180])
    upper_white = np.array([180, 40, 255])

    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_w = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_or(cv2.bitwise_or(mask_r1, mask_r2), mask_w)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_score = -1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20 or area > 3000:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        if radius < 3 or radius > 40:
            continue

        # Circularity check
        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < 0.4:
            continue

        score = circularity * area

        # Prefer detections near previous position
        if prev_pos is not None:
            dist = np.hypot(cx - prev_pos[0], cy - prev_pos[1])
            if dist > 150:
                score *= 0.3
            else:
                score *= (1 + (150 - dist) / 150)

        if score > best_score:
            best_score = score
            best = (float(cx), float(cy))

    return best


def track_ball(video_path: str, start_frame: int = 0, end_frame: Optional[int] = None) -> list[BallDetection]:
    """Track ball through video frames, return list of detections."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    detections = []
    prev_pos = None
    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if end_frame is not None and frame_idx > end_frame:
            break

        pos = detect_ball_in_frame(frame, prev_pos)
        if pos is not None:
            detections.append(BallDetection(frame=frame_idx, x=pos[0], y=pos[1], confidence=1.0))
            prev_pos = pos

        frame_idx += 1

    cap.release()
    return detections


def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    info = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info

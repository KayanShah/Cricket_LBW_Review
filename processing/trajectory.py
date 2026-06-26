import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from .ball_tracker import BallDetection


@dataclass
class TrajectoryResult:
    pre_impact_points: list[tuple[float, float]]
    post_impact_points: list[tuple[float, float]]
    stump_crossing: Optional[tuple[float, float]]
    poly_coeffs: Optional[np.ndarray]
    impact_point: tuple[float, float]
    impact_frame: int
    bounce_point: Optional[tuple[float, float]] = None


def fit_trajectory(
    detections: list[BallDetection],
    impact_point: tuple[float, float],
    bounce_point: Optional[tuple[float, float]] = None,
) -> TrajectoryResult:
    if len(detections) < 3:
        raise ValueError("Need at least 3 ball detections to fit trajectory")

    xs = np.array([d.x for d in detections])
    ys = np.array([d.y for d in detections])
    frames = np.array([d.frame for d in detections], dtype=float)

    poly_y = np.polyfit(frames, ys, 2)
    poly_x = np.polyfit(frames, xs, 1)

    dists = np.hypot(xs - impact_point[0], ys - impact_point[1])
    impact_idx = int(np.argmin(dists))
    impact_frame = int(detections[impact_idx].frame)

    pre_frames = np.linspace(frames[0], impact_frame, 40)
    pre_x = np.polyval(poly_x, pre_frames)
    pre_y = np.polyval(poly_y, pre_frames)
    pre_impact_points = list(zip(pre_x.tolist(), pre_y.tolist()))

    # Extrapolate beyond impact — same parabola (no deflection assumed for LBW)
    post_frames = np.linspace(impact_frame, impact_frame + 50, 50)
    post_x = np.polyval(poly_x, post_frames)
    post_y = np.polyval(poly_y, post_frames)
    post_impact_points = list(zip(post_x.tolist(), post_y.tolist()))

    return TrajectoryResult(
        pre_impact_points=pre_impact_points,
        post_impact_points=post_impact_points,
        stump_crossing=None,
        poly_coeffs=poly_y,
        impact_point=impact_point,
        impact_frame=impact_frame,
        bounce_point=bounce_point,
    )


def interpolate_corridor(
    point_y: float,
    stump_bottom_y: float,
    stump_left_x: float,
    stump_right_x: float,
    crease_y: Optional[float] = None,
    crease_leg_x: Optional[float] = None,
    crease_off_x: Optional[float] = None,
) -> tuple[float, float]:
    """
    Return (leg_x, off_x) — the in-line corridor boundaries at a given y.
    For face-on cameras this is just the stump box width.
    For side-on cameras with a crease reference, linearly interpolates corridor width.
    Only extrapolates if crease is meaningfully further than stumps (>30px difference).
    """
    use_perspective = (
        crease_y is not None
        and crease_leg_x is not None
        and crease_off_x is not None
        and abs(crease_y - stump_bottom_y) > 30
    )

    if not use_perspective:
        return stump_left_x, stump_right_x

    dy_total = crease_y - stump_bottom_y
    t = (point_y - stump_bottom_y) / dy_total
    leg = stump_left_x + t * (crease_leg_x - stump_left_x)
    off = stump_right_x + t * (crease_off_x - stump_right_x)
    return min(leg, off), max(leg, off)


def find_stump_closest_approach(traj: TrajectoryResult, stumps: dict) -> tuple[Optional[tuple[float, float]], float]:
    """
    Find the point on the post-impact trajectory with minimum distance to the stump box.
    Returns (closest_point, min_distance_px).
    Distance is 0 if the point is inside the box.
    """
    sx1, sx2 = stumps["left_x"], stumps["right_x"]
    sy1, sy2 = stumps["top_y"], stumps["bottom_y"]

    min_dist = float('inf')
    closest = None

    for x, y in traj.post_impact_points:
        dx = max(sx1 - x, 0.0, x - sx2)
        dy = max(sy1 - y, 0.0, y - sy2)
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist < min_dist:
            min_dist = dist
            closest = (x, y)

    return closest, min_dist


def _umpires_call_decision(cross_y: float, stump_top_y: float, stump_bottom_y: float, traj: TrajectoryResult) -> str:
    """
    Make a definitive call when the ball is clipping the stump edge.
    cross_y: pixel y where ball crosses stump x-line (screen y, increases downward).
    Returns 'OUT' or 'NOT OUT' with a short reason.
    """
    stump_h = stump_bottom_y - stump_top_y
    margin = stump_h * 0.15

    # Determine if ball is rising or falling at stump by checking post-impact trajectory slope
    pts = traj.post_impact_points
    if len(pts) >= 4:
        # y slope near stump (screen coords: positive slope = ball going downward)
        slope = pts[-1][1] - pts[len(pts)//2][1]
        going_down = slope > 0
    else:
        going_down = True  # default: assume hitting

    clipping_top = cross_y <= stump_top_y + margin

    if clipping_top:
        # Clipping top of stumps (bails level)
        if going_down:
            return "OUT"   # ball dropping into stumps
        else:
            return "NOT OUT"  # ball rising, benefit of doubt
    else:
        # Clipping side or bottom — centre still within box
        return "OUT"


def analyze_lbw(
    traj: TrajectoryResult,
    stumps: dict,
    impact_point: tuple[float, float],
    H: Optional[np.ndarray] = None,  # kept for API compat, unused
    crease_leg: Optional[tuple] = None,
    crease_off: Optional[tuple] = None,
) -> dict:
    """
    Apply LBW rules using pixel-space x comparison with optional perspective
    interpolation when a crease reference is provided and is meaningfully
    further from camera than the stumps (side-on cameras).

    For face-on cameras (most amateur footage), crease x markers are ignored
    since left-right pixels map directly to leg-off positions.
    """
    stump_left = stumps["left_x"]
    stump_right = stumps["right_x"]
    stump_bottom = stumps["bottom_y"]
    tolerance = max(5.0, (stump_right - stump_left) * 0.08)  # 8% of stump width

    crease_y = crease_leg[1] if crease_leg else None
    crease_leg_x = crease_leg[0] if crease_leg else None
    crease_off_x = crease_off[0] if crease_off else None

    def classify_x(point: tuple[float, float]) -> str:
        leg_x, off_x = interpolate_corridor(
            point[1], stump_bottom, stump_left, stump_right,
            crease_y, crease_leg_x, crease_off_x,
        )
        px = point[0]
        if px < leg_x - tolerance:
            return "OUTSIDE LEG"
        elif px > off_x + tolerance:
            return "OUTSIDE OFF"
        else:
            return "IN-LINE"

    # --- Pitching ---
    bounce_pt = traj.bounce_point
    if bounce_pt:
        pitching = classify_x(bounce_pt)
    else:
        pts = traj.pre_impact_points
        mid = pts[len(pts) // 3] if pts else impact_point
        pitching = classify_x(mid)

    # --- Impact ---
    impact = classify_x(impact_point)

    # --- Wickets ---
    stump_top = stumps["top_y"]
    stump_bottom = stumps["bottom_y"]
    # Ball radius ≈ 5% of stump height (ball 7.2cm diameter, stumps 71.1cm tall)
    # Umpire's Call = ball centre within one radius of any stump edge (<50% hitting)
    ball_radius_px = (stump_bottom - stump_top) * 0.051

    closest_pt, min_dist = find_stump_closest_approach(traj, stumps)
    traj.stump_crossing = closest_pt

    umpires_call_raw = False
    if closest_pt is None:
        wickets = "MISSING"
    elif min_dist > ball_radius_px:
        # Ball misses the box entirely — determine direction
        cx, cy = closest_pt
        if cy < stump_top:
            wickets = "MISSING OVER"
        elif cy > stump_bottom:
            wickets = "MISSING UNDER"
        else:
            wickets = "MISSING"
    else:
        # Ball is within one radius of box edge (or inside it) — check if clipping or full hit
        cx, cy = closest_pt
        sx1, sx2 = stumps["left_x"], stumps["right_x"]
        sy1, sy2 = stumps["top_y"], stumps["bottom_y"]
        # Clipping = centre is outside or within one radius of any edge
        on_edge = (cx < sx1 + ball_radius_px or cx > sx2 - ball_radius_px or
                   cy < sy1 + ball_radius_px or cy > sy2 - ball_radius_px or
                   min_dist > 0)
        if on_edge:
            wickets = "UMPIRE'S CALL"
            umpires_call_raw = True
        else:
            wickets = "HITTING"

    # --- LBW decision ---
    umpires_call_final = None

    if pitching == "OUTSIDE LEG":
        decision = "NOT OUT"
        reason = "Pitched outside leg stump — not out regardless"
    elif impact == "OUTSIDE OFF":
        decision = "NOT OUT"
        reason = "Impact outside off stump — batsman can play without risk"
    elif wickets in ("MISSING OVER", "MISSING UNDER", "MISSING"):
        decision = "NOT OUT"
        reason = "Ball missing stumps"
    elif wickets == "UMPIRE'S CALL":
        # Make a definitive call
        uc_call = _umpires_call_decision(stump_cross[1], stump_top, stump_bottom, traj)
        umpires_call_final = uc_call
        decision = uc_call
        if uc_call == "OUT":
            reason = "Umpire's Call — ball clipping stumps; trajectory dropping into stump line — OUT"
        else:
            reason = "Umpire's Call — ball barely clipping; rising trajectory, benefit of doubt to batsman — NOT OUT"
    else:
        decision = "OUT"
        reason = "Hitting stumps"

    return {
        "pitching": pitching,
        "impact": impact,
        "wickets": wickets,
        "decision": decision,
        "reason": reason,
        "umpires_call": umpires_call_raw,
        "umpires_call_verdict": umpires_call_final,
        "stump_crossing": closest_pt,
        "trajectory": {
            "pre": traj.pre_impact_points,
            "post": traj.post_impact_points,
        },
        "bounce_point": traj.bounce_point,
    }

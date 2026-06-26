import os
import json
import uuid
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename
from processing.ball_tracker import track_ball, get_video_info
from processing.trajectory import fit_trajectory, analyze_lbw

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "results")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    uid = str(uuid.uuid4())[:8]
    filename = f"{uid}_{secure_filename(f.filename)}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    f.save(path)

    try:
        info = get_video_info(path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({"video_id": uid, "filename": filename, "info": info})


@app.route("/video/<filename>")
def serve_video(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, conditional=True)


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    Expects JSON:
    {
      "video_id": "...",
      "filename": "...",
      "stumps": {"top_y", "bottom_y", "left_x", "right_x", "mid_x"},
      "impact_point": [x, y],
      "bounce_point": [x, y],          // where ball pitched
      "crease_leg": [x, y],            // leg-side end of batting crease
      "crease_off": [x, y],            // off-side end of batting crease
      "track_start_frame": 0,
      "track_end_frame": 100
    }
    """
    data = request.json
    if not data:
        return jsonify({"error": "No data"}), 400

    video_path = os.path.join(UPLOAD_FOLDER, data["filename"])
    if not os.path.exists(video_path):
        return jsonify({"error": "Video not found"}), 404

    stumps = data["stumps"]
    impact_point = tuple(data["impact_point"])
    bounce_point = tuple(data["bounce_point"]) if data.get("bounce_point") else None
    crease_leg = data.get("crease_leg")
    crease_off = data.get("crease_off")
    start_frame = data.get("track_start_frame", 0)
    end_frame = data.get("track_end_frame", None)

    try:
        detections = track_ball(video_path, start_frame, end_frame)
    except Exception as e:
        return jsonify({"error": f"Ball tracking failed: {e}"}), 500

    if len(detections) < 3:
        return jsonify({
            "error": "Not enough ball detections. Try adjusting the frame range or check video quality.",
            "detections_found": len(detections),
        }), 422

    try:
        traj = fit_trajectory(detections, impact_point, bounce_point)
    except Exception as e:
        return jsonify({"error": f"Trajectory fitting failed: {e}"}), 500

    info = get_video_info(video_path)
    result = analyze_lbw(
        traj=traj,
        stumps=stumps,
        impact_point=impact_point,
        crease_leg=tuple(crease_leg) if crease_leg else None,
        crease_off=tuple(crease_off) if crease_off else None,
    )

    result["detections"] = [{"frame": d.frame, "x": d.x, "y": d.y} for d in detections]
    result["video_width"] = info["width"]
    result["video_height"] = info["height"]
    result["perspective_correction"] = (crease_leg is not None and crease_off is not None
        and abs(crease_leg[1] - stumps["bottom_y"]) > 30)

    # Render result image
    result_img_path = _render_result_frame(video_path, result, stumps, traj, info, data["video_id"], crease_leg, crease_off)
    if result_img_path:
        result["result_image"] = f"/result_image/{os.path.basename(result_img_path)}"

    return jsonify(result)


def _render_result_frame(video_path, result, stumps, traj, info, uid, crease_leg=None, crease_off=None):
    """Render a composite result frame with trajectory overlay."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    # Use impact frame + 5 frames for a good view
    target_frame = traj.impact_frame + 5
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
        cap.release()
    if not ret:
        return None

    h, w = frame.shape[:2]

    sx1 = int(stumps["left_x"])
    sx2 = int(stumps["right_x"])
    sy1 = int(stumps["top_y"])
    sy2 = int(stumps["bottom_y"])

    # --- Stump corridor lines extending down the pitch ---
    # Determine corridor colour based on pitching result
    pitching = result.get("pitching", "IN-LINE")
    if pitching == "OUTSIDE LEG":
        corridor_colour = (255, 100, 50)   # orange-ish
    elif pitching == "OUTSIDE OFF":
        corridor_colour = (50, 200, 255)   # cyan
    else:
        corridor_colour = (60, 60, 220)    # red (in-line = danger)

    # Project corridor to frame bottom using crease markers for perspective,
    # otherwise extend vertically (face-on camera)
    if crease_leg and crease_off and abs(crease_leg[1] - sy2) > 30:
        # Side-on: taper the corridor toward crease
        t = (h - sy2) / max(crease_leg[1] - sy2, 1)
        bl_x = int(sx1 + t * (crease_leg[0] - sx1))
        br_x = int(sx2 + t * (crease_off[0] - sx2))
    else:
        # Face-on: straight vertical lines
        bl_x, br_x = sx1, sx2

    # Semi-transparent filled corridor
    overlay = frame.copy()
    corridor_pts = np.array([[sx1, sy2], [sx2, sy2], [br_x, h], [bl_x, h]], dtype=np.int32)
    cv2.fillPoly(overlay, [corridor_pts], corridor_colour)
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # Solid leg-stump and off-stump lines
    cv2.line(frame, (sx1, sy1), (bl_x, h), (180, 180, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (sx2, sy1), (br_x, h), (180, 180, 255), 2, cv2.LINE_AA)

    # Draw pre-impact trajectory (green)
    pre_pts = [(int(x), int(y)) for x, y in result["trajectory"]["pre"]]
    for i in range(1, len(pre_pts)):
        cv2.line(frame, pre_pts[i-1], pre_pts[i], (0, 200, 0), 3, cv2.LINE_AA)

    # Draw post-impact trajectory (orange extrapolation)
    post_pts = [(int(x), int(y)) for x, y in result["trajectory"]["post"]]
    for i in range(1, len(post_pts)):
        cv2.line(frame, post_pts[i-1], post_pts[i], (0, 80, 220), 3, cv2.LINE_AA)

    # Draw impact point
    ix, iy = int(traj.impact_point[0]), int(traj.impact_point[1])
    cv2.circle(frame, (ix, iy), 10, (0, 0, 255), -1)
    cv2.circle(frame, (ix, iy), 12, (255, 255, 255), 2)

    # Draw stumps box
    cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 215, 0), 2)

    # Draw stump crossing
    if result["stump_crossing"]:
        scx, scy = int(result["stump_crossing"][0]), int(result["stump_crossing"][1])
        cv2.circle(frame, (scx, scy), 12, (255, 255, 0), -1)
        cv2.circle(frame, (scx, scy), 14, (0, 0, 0), 2)

    # Draw bounce point
    if traj.bounce_point:
        bx, by = int(traj.bounce_point[0]), int(traj.bounce_point[1])
        cv2.circle(frame, (bx, by), 9, (0, 230, 120), -1)
        cv2.circle(frame, (bx, by), 11, (255, 255, 255), 2)

    # Draw crease line
    if crease_leg and crease_off:
        cl = (int(crease_leg[0]), int(crease_leg[1]))
        co = (int(crease_off[0]), int(crease_off[1]))
        cv2.line(frame, cl, co, (100, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, cl, 5, (96, 165, 250), -1)
        cv2.circle(frame, co, 5, (244, 114, 182), -1)

    out_path = os.path.join(RESULT_FOLDER, f"{uid}_result.jpg")
    cv2.imwrite(out_path, frame)
    return out_path


@app.route("/result_image/<filename>")
def serve_result(filename):
    return send_from_directory(RESULT_FOLDER, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(debug=True, port=port)

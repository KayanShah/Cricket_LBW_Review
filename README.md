<div align="center">

# 🏏 Cricket LBW Review System

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&pause=1000&color=E53935&center=true&vCenter=true&width=600&lines=Upload+a+delivery.+Mark+the+key+points.;Watch+the+trajectory+extrapolate.;Get+a+DRS-style+LBW+verdict." alt="Typing SVG" />

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **A browser-based LBW Decision Review System** — upload your cricket video, mark the stumps, bounce point, and pad impact, and get an instant DRS-style verdict with full trajectory overlay.

</div>

---

## ⚡ How It Works

```
  BOWLER                                           BATSMAN
    ○                                                 ▐█▌
    │                                                  │
    │   ●────────────────────→                         │
    │         ball in flight          ●                │
    │                              (bounce)            │
    │                                   ╲              │
    │                                    ╲─────→ ●    │
    │                                      impact ╲   ▐█
    │                                              ╲──→║ ← STUMPS?
    │                                                  ║
    ▼                                                  ▼
```

The system:
1. **Tracks the ball** frame-by-frame using HSV colour detection
2. **Fits a parabolic trajectory** to the detections
3. **Extrapolates** where the ball would have gone past the pad (assuming no deflection)
4. **Applies the real LBW rulebook** — pitching, impact, and wickets assessed independently
5. **Renders a DRS-style verdict** with four decision cards

---

## 🎬 Demo

| Step | What you do |
|------|-------------|
| **1. Upload** | Drop your MP4 / MOV / AVI (up to 500 MB) |
| **2. Mark** | Click 6 points on the video frame |
| **3. Analyze** | Hit *Analyze LBW* and wait ~10–30 s |
| **4. Result** | Full DRS card layout + trajectory overlay image |

### Decision Cards

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ ORIGINAL        │  │ WICKETS         │  │ IMPACT          │  │ PITCHING        │
│ DECISION        │  │                 │  │                 │  │                 │
│                 │  │                 │  │                 │  │                 │
│   NOT OUT       │  │   HITTING   🔴  │  │   IN-LINE   🔴  │  │   IN-LINE   🔴  │
└─────────────────┘  └─────────────────┘  └─────────────────┘  └─────────────────┘
                                    ↓
                          ╔═══════════════╗
                          ║     OUT  ❌    ║
                          ║ Hitting stumps ║
                          ╚═══════════════╝
```

---

## 🧠 LBW Rules Engine

The system implements the full LBW law:

| Check | Outcome |
|-------|---------|
| Pitched outside leg stump | **NOT OUT** — regardless of everything else |
| Impact outside off stump | **NOT OUT** — batsman free to play |
| Ball missing stumps | **NOT OUT** |
| Ball clipping stump edge (< 50% hitting) | **UMPIRE'S CALL** — verdict given |
| Ball clearly hitting | **OUT** |

### Umpire's Call Logic

When the ball is clipping the stump (ball centre within one ball radius of any stump edge), the system makes a **definitive call** rather than leaving it ambiguous:

- 🔽 Ball **dropping** into stumps at stump height → **OUT**
- 🔼 Ball **rising**, grazing bail → **NOT OUT** (benefit of doubt)

---

## 🗺️ Marking Guide

Click these 6 points on the video (the UI auto-advances between them):

```
         LEG                 OFF
          │                   │
    [1] ──┤ ← stump top  →  ├── [2] (top-right)
          ║                   ║
          ║     S T U M P S   ║
          ║                   ║
    base ─┤                   ├── [2] (bottom-right)
          │                   │
  ────────┼───────────────────┼──────  ← popping crease
    [5] ──┘                   └── [6]   (crease markers)
         crease leg         crease off


    [3] = where ball BOUNCED on pitch  (green dot)
    [4] = where ball HIT the PAD       (red dot)
```

> **Crease markers** (5 & 6) are optional but improve accuracy on side-on cameras by correcting for perspective warp. Just click the two ends of the white crease line nearest to the batsman.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install flask opencv-python numpy scipy werkzeug
```

### Run

```bash
git clone https://github.com/KayanShah/Cricket_LBW_Review.git
cd Cricket_LBW_Review
python3 app.py
```

Then open **http://localhost:5050** in your browser.

---

## 🏗️ Architecture

```
cricket-lbw/
├── app.py                    # Flask server — upload, analyze, serve results
├── processing/
│   ├── ball_tracker.py       # HSV colour detection + contour filtering
│   └── trajectory.py         # Parabola fit, extrapolation, LBW rules engine
├── templates/
│   └── index.html            # Single-page app — video player, markers, results
└── requirements.txt
```

### Ball Tracking

Detects the cricket ball using dual HSV masks (red ball + white ball) with:
- Circularity filter — rejects non-circular contours
- Area bounds — ignores noise and oversized blobs  
- Proximity weighting — prefers detections near previous frame's position

### Trajectory Model

Fits independent polynomials to detected positions:
- **X axis** — linear (`x = at + b`)
- **Y axis** — quadratic (`y = at² + bt + c`) capturing ball flight arc

Extrapolation beyond the pad impact assumes **no deflection** (the LBW assumption — where would it have gone if the pad wasn't there?).

### Wickets Detection

Rather than checking a single vertical line, the system computes the **minimum distance from every post-impact trajectory point to the full stump rectangle**:

```
dist = sqrt( max(sx1 - x, 0, x - sx2)²  +  max(sy1 - y, 0, y - sy2)² )
```

- `dist > ball_radius` → **Missing**
- `dist ≤ ball_radius`, centre near edge → **Umpire's Call**
- Centre clearly inside box → **Hitting**

---

## ⚠️ Limitations

- **Single camera only** — real Hawk-Eye uses 6+ calibrated cameras. This system reconstructs a 2D trajectory from one viewpoint.
- **Ball detection** depends on colour contrast. Dark/compressed video or non-standard ball colours may reduce detection count.
- **Face-on cameras** work best. Side-on cameras benefit from setting the crease markers for perspective correction.
- If fewer than 3 ball detections are found, try narrowing the frame range to just the delivery.

---

## 🤝 Contributing

PRs welcome. Key areas for improvement:
- YOLO-based ball detection for low-contrast footage
- Side-on camera perspective calibration
- Batsman handedness detection (right/left-hand affects leg/off sides)
- 3D trajectory reconstruction from stereo cameras

---

<div align="center">

Built with 🏏 by [KayanShah](https://github.com/KayanShah)

*Not affiliated with Hawk-Eye Innovations or the ICC*

</div>

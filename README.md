# 🛡️ Safe Turret System (AI-Powered)

> **A real-time, computer vision-based tracking and simulated turret control system.**
> *Powered by YOLOv11 Pose Estimation, MediaPipe Face Mesh, and dlib ResNet identity persistence.*

---

## 📖 Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Controls](#-controls)
- [Registering a Trusted Identity](#-registering-a-trusted-identity)
- [Modes & Aim Points](#-modes--aim-points)
- [HUD Reference](#-hud-reference)
- [Safe Zones](#-safe-zones)
- [What Gets Saved](#-what-gets-saved)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Technical Details](#-technical-details)

---

## ✨ Features

- **🎯 Keypoint-Driven Targeting** — Aims at specific body parts (forehead, chest, knees) using YOLOv11 pose keypoints, not just bbox centers. Each mode has a fallback chain that degrades gracefully when keypoints are occluded.
- **🤖 MediaPipe Face Mesh** — 478-landmark Iron Man-style HUD overlay. Runs only when a target is locked to keep the idle frame budget low.
- **🔑 Persistent Identity (PID)** — Async face encoding via dlib ResNet-34. Tracks get a stable `ID-NN` after a few frames so a person re-entering the frame keeps their identity. Registered VIPs (e.g. `COMMANDER`) take precedence with a stricter match threshold.
- **⚙️ PID-Controlled Turret** — Two-axis simulated gimbal with anti-windup integral clamping and per-target state reset.
- **🛡️ Safe Zones** — Configurable normalized rectangles where targets are visible but never selected.
- **📸 Auto-Snapshots** — When a new stable identity is seen with high confidence, the rendered frame is saved.

---

## 💻 Prerequisites

- **OS**: macOS (Apple Silicon/Intel), Windows, or Linux.
- **Python**: 3.10+ (verified on 3.11).
- **Hardware**: Webcam capable of 1920×1080 at 30 FPS. The app opens index 0 by default.
- **CPU**: Modern multi-core. Apple Silicon M-series handles 1080p at ~25–30 FPS comfortably.

---

## 🚀 Installation & Setup

### 1. Clone

```bash
git clone <repository-url>
cd turret-project
```

### 2. Virtual environment (must be named `test-env`)

The repo's `.gitignore` and CLAUDE conventions expect this exact name.

```bash
python3 -m venv test-env
source test-env/bin/activate          # macOS/Linux
# .\test-env\Scripts\activate         # Windows PowerShell
```

### 3. (Windows only) Install CMake + Visual Studio Build Tools

`dlib` ships without prebuilt wheels for some Python versions and must compile from source. See [ERROR.md](ERROR.md) for the step-by-step. macOS users need `brew install cmake boost`.

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 5. Download the MediaPipe Face Landmarker model

This is **required** before first run and is gitignored.

```bash
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

`yolo11n-pose.pt` downloads automatically on first run.

---

## 🎮 Usage

```bash
python main.py
```

A window titled **"Safe Turret Sim"** opens. **Click the window** to give it keyboard focus before pressing keys — OpenCV doesn't capture input unless its window is focused.

The app boots in **AUTO mode** with **aim mode 2 (UPPER_BODY)**.

There is no test suite, no lint config, and no headless mode. The display window must be visible for the loop to run.

---

## ⌨️ Controls

| Key   | Action                       | Notes                                                                 |
| :---: | :--------------------------- | :-------------------------------------------------------------------- |
| `1`   | Aim mode → **HEAD**          | Forehead via eyes+nose vector with multi-tier fallback.               |
| `2`   | Aim mode → **UPPER_BODY**    | Default. Shoulder midpoint + 5% bbox-height for chest.                |
| `3`   | Aim mode → **NON_LETHAL**    | Knee midpoint, falling back to hips→feet halfway.                     |
| `m`   | Toggle **AUTO ↔ MANUAL**     | Switching to manual locks onto the current primary.                   |
| `TAB` | Cycle target (manual only)   | Cycles through visible `yolo_id`s. Safe-zone targets are skipped.     |
| `r`   | **Hold ~2s** to register face | See [registration flow](#-registering-a-trusted-identity).           |
| `q`   | Quit                         | Cleanly releases camera and closes the window.                        |

Tap-and-hold `r`: each press resets a 0.5s grace window, so light tapping or steady holding both work.

---

## 🆔 Registering a Trusted Identity

Registration encodes your face into a centroid and stores it as `COMMANDER`. Once registered, any track matching that centroid (within a strict 0.5 distance) will be labeled `COMMANDER` instead of a generic `ID-NN`.

**Procedure:**

1. Stand alone in front of the camera so you are the primary target (look for the red `[ENGAGED]` bracket).
2. Hold `r`. A scanning UI overlays the screen with a progress bar.
3. Move your head **slightly left / right / up / down** during the 2-second window. The system samples ~20 face crops and averages them.
4. Watch for one of:
   - **`FINALIZING…`** (cyan): scanning is done, the encoder is still processing the last few samples. Wait 1–2 seconds.
   - **`REGISTERED`** (green): success. Console prints `[IDENTITY] Registered Trusted Identity: COMMANDER (N total samples)`.
   - **`KEEP FACE STEADY`** (red): fewer than 3 face crops were successfully encoded. Release `r`, recenter, improve lighting, try again.
5. Release `r`. Within ~1s your track's label flips from `ID-NN` to `COMMANDER` (the identity cache is invalidated on registration so existing locks re-evaluate immediately).

**Re-registration appends to the gallery** (capped at 10 samples). Subsequent holds of `r` improve match reliability rather than overwriting prior data.

**Persistence:** trusted identities and the general gallery live in memory only. They are lost on restart.

---

## 🎯 Modes & Aim Points

The active aim mode determines where the crosshair lands on each target. All modes use YOLO keypoint confidence (≥ 0.5) to gate which keypoints count as valid.

### Mode 1 — HEAD (Forehead)

1. Both eyes valid + nose valid AND eyes above nose → forehead via `eye_midpoint + 1.2 × (eye_midpoint - nose)`.
2. Both eyes valid only → `eye_midpoint - 0.8 × eye_distance` upward.
3. Nose only → 15% bbox-height above the nose.
4. No facial keypoints → top 8% of bbox.

The "eyes above nose" check protects against the head being tilted backward (which would otherwise extrapolate the aim point downward).

### Mode 2 — UPPER_BODY (Chest)

1. Both shoulders valid → shoulder midpoint + 5% bbox-height down.
2. Otherwise → 35% down from bbox top.

### Mode 3 — NON_LETHAL (Legs)

1. Both knees valid → knee midpoint.
2. Both hips valid → halfway from hips to bbox bottom.
3. Otherwise → 75% down from bbox top.

---

## 🖥️ HUD Reference

| Element                                | Meaning                                                                                       |
| -------------------------------------- | --------------------------------------------------------------------------------------------- |
| **Yellow box**                         | Tracked person, not currently locked.                                                         |
| **Red box + `[ENGAGED]`**              | Current primary lock.                                                                         |
| **Orange box + `[SAFE]`**              | In a safe zone — never selectable, even in manual mode.                                       |
| **Red dot inside each bbox**           | Computed aim point for the active aim mode.                                                   |
| **Blue dot inside each bbox**          | Safety check point (bbox center) — confirms why someone is/isn't in a safe zone.              |
| **Sci-fi crosshair (center)**          | Simulated turret aim. Cyan rotating ring when scanning, red bracket when locked.              |
| **Green line**                         | From frame center to locked target's aim point — visualizes the lock vector.                  |
| **Right panel — `TURRET CONTROL SYS`** | Pan/tilt angles, aim mode, track logic (AUTO/MANUAL), status (ENGAGED/SAFE/SCANNING).         |
| **Right panel — `ENGAGEMENT PANEL`**   | Top 5 visible targets sorted by ID with distance-to-center.                                   |
| **`TARGET LOST — TAB to reacquire`**   | Manual mode only: your selected track left the frame. Press TAB to lock onto someone else.    |

---

## 🛡️ Safe Zones

Defined in `src/config.py:SAFE_ZONES` as normalized rectangles (resolution-independent). Defaults reserve the leftmost and rightmost 15% of the frame:

```python
SAFE_ZONES = [
    (0.0, 0.0, 0.15, 1.0),  # Left edge ("Doorway")
    (0.85, 0.0, 1.0, 1.0),  # Right edge
]
```

A target whose bbox center falls inside any zone is rendered orange, labeled `[SAFE]`, and excluded from primary selection in both auto and manual modes.

---

## 💾 What Gets Saved

When a stable identity (`ID-NN` or a trusted name like `COMMANDER`) is seen with confidence > 0.6 for the first time in a session, the rendered frame is written to:

```
runs/pose/track_<unix_ts>/imageN.jpg
```

Transient `Trk-N` and `Scanning…` IDs are **not** snapshotted — this avoids the double-save that would otherwise happen when identity resolves mid-track.

The directory is created at startup; `N` is a monotonic counter. Nothing else is persisted across sessions.

---

## 📂 Project Structure

```
.
├── main.py                  # Entry point. Camera capture, main loop, HUD rendering, registration.
├── bytetrack.yaml           # ByteTrack hyperparameters for ID persistence.
├── face_landmarker.task     # MediaPipe model (gitignored — download manually).
├── yolo11n-pose.pt          # YOLOv11 pose model (gitignored — auto-downloads).
├── requirements.txt
├── runs/pose/track_<ts>/    # Per-session auto-snapshot directory (gitignored).
└── src/
    ├── config.py            # SAFE_ZONES, TURRET_LIMITS, AIM_MODES, PIXELS_PER_DEGREE.
    ├── target_manager.py    # Per-frame target list, primary selection, aim-point math.
    ├── identity_manager.py  # Async face recognition. Yolo-id → PID mapping. Trusted gallery.
    ├── turret_controller.py # PID controller with anti-windup. Pan/tilt servo simulation.
    ├── visualization.py     # draw_hud, draw_skeleton, draw_mediapipe_mesh, draw_registration_ui.
    └── skeleton_constants.py# COCO keypoint indices and skeleton edge list.
```

---

## 🔧 Troubleshooting

If installation fails or the app crashes on startup, see [ERROR.md](ERROR.md) — it covers `dlib` compilation, missing model files, and MediaPipe issues on macOS ARM64.

Common runtime issues:

- **`q` doesn't quit** — Window doesn't have keyboard focus. Click the OpenCV window first.
- **Registration always shows `KEEP FACE STEADY`** — Face too small in the bbox or poor lighting. Move closer (face ≥ ~120 px tall) and improve lighting.
- **`COMMANDER` never appears after registration** — Trusted threshold is strict (0.5). Re-register from a steady, well-lit pose; subsequent holds append to the gallery and improve match rate.
- **HUD stuttering when locked** — CPU bound. MediaPipe Face Landmarker on 1080p is the dominant cost; lower the resolution in `main.py:35-36` if needed.
- **Wrong webcam** — App opens index 0. Edit `cv2.VideoCapture(0)` in `main.py` to use a different device.

---

## 📐 Technical Details

See [TECHNICAL_DETAILS.md](TECHNICAL_DETAILS.md) for the algorithm reference: PID math, coordinate systems, async identity pipeline, and the locking-on-yolo_id rationale.

---

## 🤝 Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-thing`).
3. Commit your changes.
4. Open a Pull Request.

---

*Verified on macOS (Apple Silicon, M-series) with Python 3.11.*

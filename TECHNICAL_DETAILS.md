# Technical Details

This document covers the math, data flow, and design decisions that aren't obvious from reading the code linearly.

---

## Per-Frame Pipeline

```
cv2.VideoCapture
    ↓ frame (BGR, 1920×1080)
YOLOv11 Pose + ByteTrack       ← model.track(frame, persist=True, classes=[0])
    ↓ boxes + 17 keypoints + transient yolo_id
TargetManager.select_targets   ← per-target safety check + aim point + identity lookup
    ↓ list[target_data], primary
[capture primary_crop, mp_rgb] ← BEFORE drawing, so AI consumers get clean pixels
TurretController.update        ← pixel error → pan/tilt PID
    ↓
draw_hud → draw_skeleton → draw_mediapipe_mesh   (drawing on `frame`)
    ↓
auto-snapshot block            ← writes only stable IDs (ID-NN, COMMANDER, ...)
    ↓
cv2.imshow + cv2.waitKey
```

The `frame` buffer is **not copied** for AI consumers. The pipeline relies on the fact that YOLO and `IdentityManager` run *before* any HUD drawing, and `IdentityManager` crops + copies internally before submitting to its background thread. The two consumers that need clean pixels *after* `select_targets` returns — registration encoding and MediaPipe — read from `primary_crop` (face-zone slice) and `mp_rgb` (full-frame BGR→RGB conversion), both captured before `draw_hud`.

---

## YOLO Pose & Tracking

- Model: `yolo11n-pose.pt` (Ultralytics nano variant). Returns 17 COCO keypoints per person plus per-keypoint confidence.
- Tracker: ByteTrack via `bytetrack.yaml`. Defaults: `track_buffer=30` (≈1 second of buffer for occluded targets), `match_thresh=0.8`, `new_track_thresh=0.25`.
- `model.track(...)` is invoked with `verbose=False, classes=[0]` and **no disk-write flags**. Selective snapshots are handled by the auto-snapshot block, not by Ultralytics' internal `save=True`.

### Why YOLO `yolo_id`s are transient

ByteTrack's IDs are stable within a continuous track but reset when a person is occluded longer than `track_buffer`. This is why locking is anchored on `yolo_id` for the **current frame**, while the user-facing identity comes from the async PID system.

---

## Aim-Point Computation

For each target, `_compute_aim_point()` (in `target_manager.py`) returns a precise pixel target based on the active aim mode. All keypoint validity checks use the **confidence channel** (`keypoints.data[:, :, 2] >= 0.5`) rather than `(x, y) > 0` — this avoids treating frame-edge detections as missing.

### Mode 1 — HEAD

```
if eyes_valid and nose_valid and eyes_above_nose:
    aim = eye_midpoint + 1.2 × (eye_midpoint - nose)
elif eyes_valid:
    aim = eye_midpoint + (0, -0.8 × eye_distance)
elif nose_valid:
    aim = nose + (0, -0.15 × bbox_height)
else:
    aim_y = bbox_top + 0.08 × bbox_height
```

The "eyes above nose" guard prevents extrapolation in the wrong direction when the head is tilted backward.

### Mode 2 — UPPER_BODY

```
if shoulders_valid:
    aim = shoulder_midpoint + (0, 0.05 × bbox_height)
else:
    aim_y = bbox_top + 0.35 × bbox_height
```

The 5% offset is a fixed bbox-height fraction. At close range it under-shoots the chest center (chest is closer to 15–20% below the shoulder line), at distance it's roughly correct.

### Mode 3 — NON_LETHAL

```
if knees_valid:
    aim = knee_midpoint
elif hips_valid:
    aim = hip_midpoint + (0, 0.5 × (bbox_bottom - hip_midpoint_y))
else:
    aim_y = bbox_top + 0.75 × bbox_height
```

---

## PID Turret Controller

`TurretController.update(error_x, error_y)` runs once per frame with the pixel error (target position minus frame center).

### Standard PID with anti-windup

```
integral_x = clip(integral_x + error_x, -INTEGRAL_CLAMP, +INTEGRAL_CLAMP)
derivative_x = error_x - prev_error_x
output_x = kp·error_x + ki·integral_x + kd·derivative_x
```

`INTEGRAL_CLAMP = 500` is critical. Without it, a long off-center lock (e.g. 600 px error for 3 seconds at 30 FPS) would push the integral term to ~108,000, producing a multi-second overshoot after the target is centered. With the clamp, the integral contributes at most `ki × 500 = 5°` per update before output clipping.

### Output transform

```
delta_pan  = clip(0.1 × output_x, ±MAX_PAN_SPEED)
delta_tilt = clip(0.1 × output_y, ±MAX_TILT_SPEED)
pan_angle  = clip(pan_angle + delta_pan,  ±90°)
tilt_angle = clip(tilt_angle - delta_tilt, ±45°)
```

Note the **subtraction** for `tilt_angle`. This is intentional, not a bug: positive pixel-Y means the target is *below* the frame center, so the camera needs to look *down*, which is a *negative* tilt angle in our convention. The block comment in `get_current_aim_point` documents the verification math.

### Reset on lock change

`TargetManager._update_primary` sets `target_lost = True` for one frame whenever the locked `yolo_id` changes (or drops to None). The main loop calls `turret.reset_state()` on that frame, zeroing all PID memory. Without this reset, the integral and derivative terms would carry over from the previous target — causing a sudden snap when TAB-cycling between people.

### Coordinate conventions

- **Frame**: top-left is (0, 0); +X right, +Y down.
- **Pan**: 0° centered; +° right, -° left.
- **Tilt**: 0° centered; +° up, -° down (inverted from pixel Y).
- **Calibration**: `PIXELS_PER_DEGREE = 32.0`, valid for 1920px width / ~60° FOV. Recalibrate if you change capture resolution or lens.

---

## Identity Resolution (`IdentityManager`)

The async identity layer maps transient `yolo_id`s to persistent identifiers. **The main thread is never blocked by `face_recognition`.**

### Data structures

```
known_entities      = { 'ID-01': {encodings: [...], created_at, last_seen}, ... }
trusted_identities  = { 'COMMANDER': [encoding1, encoding2, ...] }
yolo_to_pid         = { yolo_id: 'ID-01' | 'COMMANDER' | 'Trk-7' | 'Scanning...' }
last_check_time     = { yolo_id: timestamp }
confirm_counts      = { yolo_id: int }
confirmed_ids       = set of yolo_ids whose PID is locked in
pending_tasks       = set of yolo_ids currently being processed
```

### `get_pid(frame, box, yolo_id)` — non-blocking

1. If `yolo_id ∈ confirmed_ids` → return cached PID immediately. **No further face_recognition calls** for this track.
2. If a recent check is still within `check_interval` (1.0s) → return cached PID.
3. If already in `pending_tasks` → return cached or `"Scanning..."`.
4. If `len(pending_tasks) >= max_inflight` (4) → return cached or `"Trk-{yolo_id}"`. Backpressure prevents the executor queue from growing without bound under heavy multi-person load.
5. Otherwise: crop the **face zone** (top 40% of bbox, min 60px), `.copy()` it, submit to executor, return cached or `"Scanning..."`.

### Background worker `_process_face_bg`

1. Resize crop to 200px width if larger (face_recognition is O(N²) in pixel count).
2. `face_recognition.face_encodings(rgb)` — HOG detector + ResNet-34 encoding.
3. `_match_encoding`:
   - **Trusted gallery first** with `trusted_tolerance = 0.5` (strict).
   - **General gallery** with `match_tolerance = 0.6` (face_recognition's standard).
   - Returns the closest match below threshold, or `(None, None)` for a new face.
4. If new → assign `ID-{counter:02d}`, start gallery with this encoding.
5. If matched → optionally augment the gallery (cap 5 encodings/PID for online learning).
6. Update `yolo_to_pid[yolo_id]` and `confirm_counts`. After `confirm_threshold = 3` consecutive matches to the same PID, the yolo_id joins `confirmed_ids` and never re-checks.

### Why face-zone cropping matters

The bbox from YOLO covers the entire person (head + torso). face_recognition's HOG detector finds the face inside that crop, but at small face-pixel sizes detection is noisy. Cropping the **top 40% of the bbox** before encoding gives the detector a face-centric image — used both during registration and during identity scans, so encodings are comparable.

### Trusted Identity Registration

`register_trusted_identity(name, encodings)`:

1. Appends the new encodings to existing samples for `name` (capped at 10). **Re-registration augments, doesn't clobber.**
2. Clears `confirmed_ids`, `confirm_counts`, and `last_check_time`. This forces all currently-tracked yolo_ids to re-evaluate their PID — necessary because a yolo_id confirmed as `ID-03` *before* the COMMANDER registration would otherwise stay frozen on the old PID forever.

### Why the trusted threshold is stricter

A false `ID-NN` match causes the same person to share an ID — annoying but recoverable. A false `COMMANDER` match means the system trusts a stranger as a registered VIP. Asymmetric cost ⇒ asymmetric thresholds: 0.5 for trusted, 0.6 for general.

---

## Locking & Selection

Locking anchors on **`yolo_id`** (transient but stable within a track), not on PID:

- `TargetManager.selected_yolo_id` — the user's TAB selection in manual mode.
- `TargetManager.last_primary_yolo_id` + `target_lost` — set by `_update_primary()` so the main loop can reset the PID controller exactly when the lock changes.
- `'locked'` field on each target dict is rebuilt every frame (only the primary gets it). There is no persistent `locked_ids` set.

### Why not store the user's selection as a PID?

PIDs are async-resolved. A track flips from `Trk-7` → `ID-03` → `COMMANDER` over the first few frames. If `selected_id` were a PID string, the user's TAB selection would silently break each time the underlying PID changed. Anchoring on `yolo_id` keeps the lock stable through identity promotion.

The downside: ByteTrack's `track_buffer = 30` means a person occluded for more than ~1 second gets a new `yolo_id` and the lock breaks. The face-recognition layer eventually stitches them back to the same PID, but the `yolo_id` jump is visible.

---

## Auto-Snapshots

When a target with a **stable PID** (anything not `Trk-N` or `Scanning...`) is seen for the first time in the session with `conf > 0.6`, the *rendered* frame is written to `runs/pose/track_<unix_ts>/imageN.jpg`.

- Transient `Trk-N` IDs are excluded — they would cause double snapshots when identity later resolves to `ID-NN`.
- Snapshot dedup is keyed on the PID string itself; once `ID-03` is snapshotted, it won't snapshot again until process restart.
- The directory is created once at startup; `snapshot_counter` is a single-element list (mutable closure) for the filename index, replacing the previous per-frame `os.listdir` scan.

---

## Facial Visualization (MediaPipe Face Landmarker)

- **API**: MediaPipe Tasks API (`vision.FaceLandmarker.detect_for_video`), not the legacy `mp.solutions.face_mesh`. The Tasks API is more robust on macOS ARM64 — see ERROR.md.
- **Mesh**: 478 3D landmarks (face contour + iris + lips + eyebrows).
- **Frame budget**: gated on `primary is not None`. Running the mesh on a 1920×1080 frame every iteration is the dominant idle CPU cost; gating it cuts the idle frame time substantially.
- **Timestamp**: monotonic milliseconds from `start_time_s`. The Tasks API video mode requires monotonically increasing timestamps.

---

## Registration Flow (User-Facing State Machine)

The registration loop lives in `main.py`, not `IdentityManager`. It is a small state machine with these variables:

| Variable                     | Role                                                                          |
| ---------------------------- | ----------------------------------------------------------------------------- |
| `reg_start_time`             | When the current registration hold began. `None` when idle.                   |
| `last_r_time`                | Most recent `r` keypress. The 0.5s grace window is computed from this.        |
| `accumulated_encodings`      | Successfully encoded face crops since `reg_start_time`.                       |
| `pending_encoding_futures`   | In-flight `Future`s from `IdentityManager.encode_async`.                      |
| `registration_done`          | Latched True after a successful register; reset on R release.                 |

### Sequence

1. User presses `r` → `last_r_time` and `reg_start_time` are set; buffers cleared.
2. Each frame while `is_registering`:
   - Drain any `Future` that has `.done()` into `accumulated_encodings`.
   - Every 0.1s, submit a new `primary_crop` to `encode_async()`. Encoding runs on the same single-worker executor that powers `get_pid` — they share the queue.
   - `progress = duration / 2.0` drives the on-screen progress bar.
3. At `progress >= 1.0`:
   - Pending futures present → show **`FINALIZING…`** and wait.
   - Otherwise, ≥ 3 encodings collected → average them into a centroid, call `register_trusted_identity("COMMANDER", [centroid])`, set `registration_done = True`.
   - Otherwise → show **`KEEP FACE STEADY`** (insufficient samples).
4. While `registration_done` is True (still holding `r`) → show **`REGISTERED`** in green.
5. User releases `r` → after 0.5s grace, `is_registering` drops to False (and stays True only if futures are still pending). All buffers reset.

### Critical timing detail

`is_registering` is extended past the 0.5s grace window if futures are still pending. Without this extension, a user releasing `r` exactly at `progress == 1.0` could exit the registration block with futures still in flight — and the late-arriving encoding would simply be discarded.

---

## Dependencies

- **OpenCV** (`opencv-python`) — Camera capture, BGR↔RGB, HUD drawing.
- **NumPy** — Vector math, PID arithmetic, encoding centroid computation.
- **Ultralytics** (`ultralytics`) — YOLOv11 inference + ByteTrack.
- **MediaPipe** — Face Landmarker (Tasks API).
- **face_recognition** + **dlib** — ResNet-34 face encoding + HOG detector.

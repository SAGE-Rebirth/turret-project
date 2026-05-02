# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Run / Setup

```bash
# venv (the repo expects this exact name; .gitignore excludes it)
python3 -m venv test-env && source test-env/bin/activate
pip install -r requirements.txt

# face_landmarker.task is gitignored and MUST be present at repo root before running
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# yolo11n-pose.pt downloads automatically on first run (also gitignored)
python main.py
```

There is no test suite, no linter config, and no build step — `python main.py` is the only entry point. It opens the default webcam at 1920x1080 and an OpenCV display window; it cannot run headless.

Runtime keys: `1/2/3` (Head/Upper-Body/Legs aim), `m` (manual ↔ auto), `TAB` (cycle target in manual), `r` (hold ~2s to register face as `COMMANDER`), `q` (quit).

## Architecture

Single-process pipeline driven by `main.py`'s while loop. Every frame:

1. **Capture** — `frame` is read once. There is no full-frame copy: YOLO and `IdentityManager` run *before* any HUD drawing, so they see a clean image. `IdentityManager` then crops + `.copy()`s before handing data to its thread. Two AI consumers that need clean pixels *after* `select_targets` returns — registration encoding and MediaPipe — read from `primary_crop` and `mp_rgb`, which are captured at line ~135 *before* `draw_hud`. Don't move those captures past `draw_hud` or label rectangles will bleed into face encoding/mesh detection.
2. **YOLO + ByteTrack** (`yolo11n-pose.pt`, `bytetrack.yaml`) emits transient `yolo_id`s plus 17 COCO keypoints. `model.track(...)` is called with `verbose=False, classes=[0]` only — disk-write flags (`save`, `save_txt`, `save_conf`) are deliberately off; selective snapshots are written by the auto-snapshot block instead.
3. **`TargetManager.select_targets`** (`src/target_manager.py`) computes per-target `aim_point` from keypoints with confidence ≥ `KP_CONF_THRESHOLD = 0.5` (uses `keypoints.data` channel 3, not `(x,y) > 0`):
   - mode 1 (HEAD): nose-through-eyes vector × 1.2 to forehead, *only when eyes are above nose* (head not tilted back); else falls back to eye-distance × 0.8
   - mode 2 (UPPER_BODY): shoulder midpoint + 5% bbox-height down for chest
   - mode 3 (NON_LETHAL): knee midpoint, then hips→feet halfway, then 75% bbox height
   Aim point math is in `_compute_aim_point()`; preserve the fallback chains.
4. **Safe zones** (`src/config.py:SAFE_ZONES`) are normalized rectangles. Targets inside them are kept in `valid_targets` with `safe=True` but excluded from primary selection (and locked out in manual mode too). Don't filter them earlier — the HUD renders them as "SAFE" in orange.
5. **Identity resolution** — `IdentityManager.get_pid` (`src/identity_manager.py`) is non-blocking. It returns a cached `ID-NN` / `Trk-<yolo_id>` / `Scanning…` string and may submit a `face_recognition.face_encodings` job to a single-worker `ThreadPoolExecutor`. Three rate limits matter:
   - `check_interval = 1.0s` between re-checks per `yolo_id`
   - `confirm_threshold = 3` consecutive matches → the `yolo_id` joins `confirmed_ids` and never re-checks again
   - `max_inflight = 4` cap; new submissions over cap return cached fallback rather than queuing
   `register_trusted_identity` averages multi-sample encodings into a centroid; `_match_encoding` checks trusted identities first, then the general gallery (capped at 5 encodings/PID for online learning). `encode_async(bgr_crop)` is exposed for the registration flow to share the same executor.
6. **PID** — `TurretController.update` (`src/turret_controller.py`) takes pixel error and returns clipped pan/tilt angles. Two non-obvious things:
   - Integral term is clamped per-update to `±INTEGRAL_CLAMP = 500` (anti-windup). Don't remove this — without it a long off-center lock causes seconds of overshoot.
   - The tilt sign convention is intentional: `tilt = tilt - delta_tilt` because positive pixel-Y (target below center) must produce a negative tilt angle (look down). Block comment in `get_current_aim_point` documents the math — do not "fix" the inverted sign.
   - `reset_state()` zeros all PID memory. The main loop calls it whenever `manager.target_lost` fires (lock changed or dropped) so the integral/derivative don't reference a prior target.
7. **Render** — `draw_hud` then `draw_skeleton` (per-person from raw YOLO keypoints) then `draw_mediapipe_mesh` (MediaPipe Tasks API `FaceLandmarker.detect_for_video`, called with monotonic ms timestamp from `start_time_s`). MediaPipe runs **only when `primary is not None`** — full-frame mesh is the dominant CPU cost when idle. Tasks API is used deliberately instead of legacy `mp.solutions` — see `ERROR.md`.
8. **Auto-snapshot** — first time a stable PID (string starting with `ID-`) is seen with `conf > 0.6`, the rendered frame is written to `runs/pose/track_<unix_ts>/imageN.jpg`. Transient `Trk-N` and `Scanning...` IDs are skipped to avoid double snapshots when identity resolves. The directory is created once at startup; `snapshot_counter` is a single-element list (mutable closure) for the filename index.

## Locking / selection state (read this before changing TAB or `m`)

Locks anchor on `yolo_id` (transient but stable within a track), **not** on PID. PIDs are async-resolved and a track flips from `Trk-7` → `ID-03` mid-session; if `selected_id` were a PID the user's TAB selection would silently break. Concretely:

- `TargetManager.selected_yolo_id` — the user's chosen track in manual mode.
- `TargetManager.last_primary_yolo_id` + `target_lost` — set by `_update_primary()` so `main.py` can call `turret.reset_state()` exactly when the lock changes.
- `'locked'` field on each target dict is rebuilt every frame (only the primary gets it). There is no persistent `locked_ids` set.
- Manual + selected track gone-from-frame → `primary` is `None` and the HUD shows "TARGET LOST — TAB to reacquire". Auto mode picks closest-to-center.

## Cross-file invariants

- `main.py` reads `key` from the *previous* iteration's `cv2.waitKey` (set at loop end, consumed at top). Refactors that move the read break input handling.
- The registration state machine (`reg_start_time`, `last_r_time`, 0.5s grace + 2s duration + ≥3 samples + `pending_encoding_futures`) lives in `main.py`, not `IdentityManager`. Encoding is offloaded via `id_manager.encode_async()` so the display never blocks; futures are drained non-blocking each frame.
- COCO keypoint indices are documented in `src/skeleton_constants.py`. Targeting code references them by raw index (0=nose, 1/2=eyes, 5/6=shoulders, 11/12=hips, 13/14=knees) — keep that file in sync if you change indexing.
- `PIXELS_PER_DEGREE = 32.0` in `config.py` is calibrated for 1920px / ~60° FOV. Recalibrate if you change capture resolution.

## Platform notes

- `dlib` (used by `face_recognition`) compiles from source on Windows — needs CMake on PATH and Visual Studio C++ Build Tools. macOS needs `brew install cmake boost`. `ERROR.md` is the canonical reference.
- Verified target: macOS (Apple Silicon) + Python 3.11. MediaPipe on macOS ARM64 has occasionally shipped without `mp.solutions` lazy-loaded; this repo migrated to the Tasks API to avoid that class of bugs — don't reintroduce `import mediapipe.solutions` calls.

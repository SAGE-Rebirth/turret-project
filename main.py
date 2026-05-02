import os
import cv2
import numpy as np
import time
try:
    from ultralytics import YOLO
except ImportError:
    print("\n[ERROR] 'ultralytics' library not found.")
    print("Please run: pip install ultralytics")
    print("Then try running this script again.\n")
    exit(1)

from src.turret_controller import TurretController
from src.target_manager import TargetManager
from src.visualization import draw_hud, draw_registration_ui, draw_skeleton, draw_mediapipe_mesh

def main():
    print("[SYSTEM] Initializing Safe Turret System...")
    
    # Load POSE Model for precise keypoint targeting
    try:
        model = YOLO("yolo11n-pose.pt") 
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Camera not found.")
        return

    # Set resolution to 1920x1080 (Full HD)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Wait for the first frame to ensure we have dimensions
    time.sleep(2) # Allow camera to warm up
    ret, test_frame = cap.read()
    if not ret:
        print("[ERROR] Could not read from camera.")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Init Subsystems
    turret = TurretController(kp=0.1, ki=0.01, kd=0.05)
    manager = TargetManager(W, H)
    
    # Initialize MediaPipe Tasks API (New Face Landmarker)
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    
    # Path to the downloaded model
    model_path = 'face_landmarker.task'
    
    # Create options
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True, # Optional: For detailed expressions
        output_facial_transformation_matrixes=True,
        num_faces=5, # Track up to 5 faces simultaneously
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO)
    
    # Create Landmarker
    landmarker = vision.FaceLandmarker.create_from_options(options)
    
    aim_mode = 2 # Default: UPPER_BODY
    
    print(f"[SYSTEM] Cam: {W}x{H}")
    print("[SYSTEM] Mode: PRECISE POSE TRACKING + MP TASKS API")
    print("[CONTROL] Keys: '1'=Head, '2'=Upper Body, '3'=Non-Lethal")
    print("[CONTROL] Keys: 'm'=Toggle Manual/Auto, 'TAB'=Cycle Targets")

    # Timestamp for MP (monotonic in ms)
    # import time (Removed: Global import used)
    start_time_s = time.time()
    
    # Registration State
    reg_start_time = None
    last_r_time = 0
    key = -1 
    
    # Multi-Sample State
    accumulated_encodings = []
    pending_encoding_futures = []
    last_sample_time = 0
    registration_done = False  # latched True after a successful register
                               # within one R-hold; reset on release
    
    # Snapshot State
    session_snapshots = set()
    snapshot_dir = os.path.join("runs", "pose", f"track_{int(time.time())}")
    os.makedirs(snapshot_dir, exist_ok=True)
    snapshot_counter = [0]  # mutable so the inner block can increment

    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("[ERROR] Camera frame dropped.")
                break
            
            # Frame ownership: YOLO + IdentityManager run BEFORE any HUD
            # drawing, so they can read `frame` directly. IdentityManager
            # crops + copies internally before handing data to its thread.
            ts_ms = int((time.time() - start_time_s) * 1000)

            try:
                results = model.track(frame, persist=True, tracker="bytetrack.yaml",
                                    verbose=False, classes=[0])
            except Exception as e:
                print(f"[ERROR] YOLO Track Failed: {e}")
                results = None

            targets = []
            primary = None

            if results:
                targets = manager.select_targets(results[0], frame, aim_mode)
                primary = manager.primary_target

            # Snapshot crops BEFORE any HUD drawing so AI consumers
            # (registration encoding, MediaPipe mesh) see a clean image.
            # primary_crop targets the TOP 40% of the bbox so face_recognition's
            # HOG detector gets a face-centric crop instead of a small face
            # inside a tall body crop — improves encoding quality during
            # registration and identity scans.
            primary_crop = None
            if primary is not None:
                px1, py1, px2, py2 = primary['box']
                px1, py1 = max(0, px1), max(0, py1)
                px2, py2 = min(W, px2), min(H, py2)
                if px2 > px1 and py2 > py1:
                    bh = py2 - py1
                    face_y2 = py1 + max(int(bh * 0.4), 60)
                    face_y2 = min(face_y2, py2)
                    primary_crop = frame[py1:face_y2, px1:px2].copy()

            mp_rgb = None
            if primary is not None:
                mp_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
            # --- TURRET PID UPDATE ---
            # Reset PID memory whenever the locked track changes (or we
            # lose the lock) so the integral/derivative don't reference a
            # prior target's error history.
            if manager.target_lost:
                turret.reset_state()

            if primary:
                target_x, target_y = primary['aim_point']
                error_x = target_x - (W // 2)
                error_y = target_y - (H // 2)
                turret.update(error_x, error_y)
            
            # --- INPUT LOGIC (Applied to CURRENT frame) ---
            # Note: 'key' comes from the PREVIOUS iteration's waitKey
            if key == ord('q'): 
                break
            elif key == ord('1'):
                aim_mode = 1
            elif key == ord('2'):
                aim_mode = 2
            elif key == ord('3'):
                aim_mode = 3
            elif key == ord('m'):
                manager.manual_mode = not manager.manual_mode
                print(f"[SYSTEM] Manual Mode: {manager.manual_mode}")
                if manager.manual_mode and manager.primary_target:
                    manager.selected_yolo_id = manager.primary_target['yolo_id']
                elif not manager.manual_mode:
                    manager.selected_yolo_id = None
            elif key == 9:  # TAB
                if manager.manual_mode and len(targets) > 0:
                    yolo_ids = sorted(t['yolo_id'] for t in targets)
                    if manager.selected_yolo_id in yolo_ids:
                        idx = yolo_ids.index(manager.selected_yolo_id)
                        manager.selected_yolo_id = yolo_ids[(idx + 1) % len(yolo_ids)]
                    else:
                        manager.selected_yolo_id = yolo_ids[0]
                    print(f"[SYSTEM] Switched Target -> yolo_id={manager.selected_yolo_id} "
                          f"(Available: {yolo_ids})")
    
            # --- REGISTRATION LOGIC (Robust & Multi-Sample) ---
            if key == ord('r'):
                last_r_time = time.time()
                if reg_start_time is None:
                    reg_start_time = time.time()
                    accumulated_encodings = []
                    pending_encoding_futures = []

            # Stay in registering state if futures are still pending so a
            # late-arriving encoding can still complete registration even
            # after the user releases R.
            is_registering = (time.time() - last_r_time < 0.5
                              or (reg_start_time is not None
                                  and pending_encoding_futures))

            if is_registering and reg_start_time is not None:
                duration = time.time() - reg_start_time
                progress = min(1.0, duration / 2.0)
                draw_registration_ui(frame, progress)

                # Drain finished encoding futures (non-blocking).
                still_pending = []
                for fut in pending_encoding_futures:
                    if fut.done():
                        enc = fut.result()
                        if enc is not None:
                            accumulated_encodings.append(enc)
                    else:
                        still_pending.append(fut)
                pending_encoding_futures = still_pending

                # Submit a new sample every 0.1s — encoding runs on the
                # IdentityManager's executor, never blocks the display.
                if primary and primary_crop is not None and progress < 1.0 \
                        and time.time() - last_sample_time > 0.1:
                    pending_encoding_futures.append(
                        manager.id_manager.encode_async(primary_crop)
                    )
                    last_sample_time = time.time()

                if progress >= 1.0:
                    # Wait briefly for any in-flight encodings to land before
                    # deciding success/failure. Encoding takes ~50-100ms each
                    # and the executor is single-worker, so a 2s window can
                    # leave several futures still pending at progress==1.0.
                    if pending_encoding_futures:
                        cv2.putText(frame, "FINALIZING…",
                                    (W // 2 - 80, H // 2 + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    elif registration_done:
                        # Already registered this hold — show success until
                        # the user releases R.
                        cv2.putText(frame, "REGISTERED",
                                    (W // 2 - 90, H // 2 + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif len(accumulated_encodings) >= 3:
                        mean_encoding = np.mean(accumulated_encodings, axis=0)
                        manager.id_manager.register_trusted_identity(
                            "COMMANDER", [mean_encoding]
                        )
                        print(f"[SYSTEM] Registered COMMANDER with "
                              f"{len(accumulated_encodings)} samples.")
                        registration_done = True
                    else:
                        cv2.putText(frame, "KEEP FACE STEADY",
                                    (W // 2 - 100, H // 2 + 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                reg_start_time = None
                accumulated_encodings = []
                pending_encoding_futures = []
                registration_done = False
    
            # --- RENDER LAYERS ---
            draw_hud(frame, turret, targets, primary, aim_mode, manager)

            # Manual-mode lock-lost banner: user TAB'd onto a track that
            # is no longer visible (occluded, off-screen, ID-switched).
            if manager.manual_mode and manager.selected_yolo_id is not None and primary is None:
                cv2.putText(frame, "TARGET LOST — TAB to reacquire",
                            (W // 2 - 260, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
            # --- TECH DEMO VISUALS ---
            # 1. Pose Skeleton (For ALL tracked persons)
            if results and results[0].keypoints is not None:
                 for i, kps in enumerate(results[0].keypoints):
                     kp_xy = kps.xy[0].cpu().numpy()
                     draw_skeleton(frame, kp_xy)

            # 2. MediaPipe Face Landmarker (Tasks API) — only when locked.
            # mp_rgb was captured BEFORE any HUD drawing so labels/boxes
            # don't bleed into the mesh.
            if mp_rgb is not None:
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=mp_rgb)
                detection_result = landmarker.detect_for_video(mp_image, ts_ms)
                if detection_result.face_landmarks:
                    draw_mediapipe_mesh(frame, detection_result.face_landmarks)

            # --- 4. AUTO-SNAPSHOT (After Rendering) ---
            # Only snapshot stable PIDs (ID-NN or trusted names like COMMANDER).
            # Transient "Trk-N" / "Scanning..." would otherwise cause double
            # snapshots once identity resolves.
            if results and snapshot_dir:
                new_ids_this_frame = []
                for t in targets:
                    tid = t['id']
                    if not isinstance(tid, str):
                        continue
                    if tid.startswith("Trk-") or tid.startswith("Scanning"):
                        continue
                    conf = t.get('conf', 0.0)
                    if tid not in session_snapshots and conf > 0.6:
                        new_ids_this_frame.append(tid)
                        session_snapshots.add(tid)

                if new_ids_this_frame:
                    filename = f"image{snapshot_counter[0]}.jpg"
                    snapshot_counter[0] += 1
                    path = os.path.join(snapshot_dir, filename)
                    cv2.imwrite(path, frame)
                    print(f"[SYSTEM] Snapshot Saved: {path} (Trigger: {new_ids_this_frame})")
    
            # --- DISPLAY ---
            cv2.imshow("Safe Turret Sim", frame)
            
            # --- UPDATE KEY ---
            key = cv2.waitKey(1)
            
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Main loop crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

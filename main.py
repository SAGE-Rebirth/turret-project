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
from src.visualization import draw_hud, draw_registration_ui

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
    last_sample_time = 0
    
    # Snapshot State
    session_snapshots = set()

    try:
        while True:
            ret, frame = cap.read()
            if not ret: 
                print("[ERROR] Camera frame dropped.")
                break
            
            # --- FRAME SAFETY ---
            # Create a clean copy for AI/Threads to use. 
            # This prevents race conditions where 'frame' is modified by drawing whilst AI reads it.
            clean_frame = frame.copy()
            
            # Calculate MS timestamp
            ts_ms = int((time.time() - start_time_s) * 1000)
    
            # Track with ByteTrack (Use CLEAN frame)
            # Enable logging: save_txt=True, save_conf=True
            try:
                results = model.track(clean_frame, persist=True, tracker="bytetrack.yaml", 
                                    verbose=False, classes=[0],
                                    save=True,      # Save inference images/video
                                    save_txt=True,  # Save bounding box coordinates
                                    save_conf=True  # Save confidence scores
                                    )
            except Exception as e:
                print(f"[ERROR] YOLO Track Failed: {e}")
                results = None
            
            # --- LOGIC UPDATE ---
            targets = []
            primary = None
            
            if results:
                run_dir = results[0].save_dir # Get YOLO's active run folder
                
                # Pass the CLEAN results and CLEAN frame for identity
                targets = manager.select_targets(results[0], clean_frame, aim_mode)
                primary = manager.primary_target
                
            # --- TURRET PID UPDATE ---
            if primary:
                # Use the calculated AIM POINT
                target_x, target_y = primary['aim_point']
                
                # Error = Target Point - Frame Center
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
                # If switching to Manual, lock onto current primary (if any)
                if manager.manual_mode and manager.primary_target:
                    manager.selected_id = manager.primary_target['id']
                elif not manager.manual_mode:
                    manager.selected_id = None
            elif key == 9: # TAB Key (ASCII 9)
                if manager.manual_mode and len(targets) > 0:
                    # Cycle ID
                    # Get list of IDs
                    ids = sorted([t['id'] for t in targets])
                    
                    if manager.selected_id in ids:
                        idx = ids.index(manager.selected_id)
                        next_idx = (idx + 1) % len(ids)
                        manager.selected_id = ids[next_idx]
                    else:
                        # If current selected is lost, pick first
                        manager.selected_id = ids[0]
                    
                    print(f"[SYSTEM] Switched Target -> {manager.selected_id} (Available: {ids})")
    
            # --- REGISTRATION LOGIC (Robust & Multi-Sample) ---
            # Update State based on Key
            if key == ord('r'):
                last_r_time = time.time()
                if reg_start_time is None:
                    reg_start_time = time.time()
                    accumulated_encodings = [] # Reset buffer
            
            # Check if we are in "Active" state (Grace period 0.5s)
            is_registering = (time.time() - last_r_time < 0.5)
            
            if is_registering and reg_start_time is not None:
                 duration = time.time() - reg_start_time
                 progress = min(1.0, duration / 2.0)
                 
                 # Draw UI Overlay
                 draw_registration_ui(frame, progress)
                 
                 # --- ALGORITHMIC SCANNING (Multi-Sample) ---
                 # Every 0.1s, try to capture a sample
                 if primary and progress < 1.0:
                     if time.time() - last_sample_time > 0.1:
                         # Sample Face
                         x1, y1, x2, y2 = primary['box']
                         x1, y1 = max(0, x1), max(0, y1)
                         x2, y2 = min(W, x2), min(H, y2)
                         
                         # Use CLEAN frame for data
                         # Make a copy of the frame before drawing on it for clean crop
                         # Note: clean_frame is already clean
                         crop = clean_frame[y1:y2, x1:x2]
                         if crop.size > 0:
                             import face_recognition
                             rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                             encs = face_recognition.face_encodings(rgb_crop)
                             if len(encs) > 0:
                                 accumulated_encodings.append(encs[0])
                                 # feedback (optional print)
                                 # print(f".", end="", flush=True) 
                             
                         last_sample_time = time.time()
    
                 # Trigger Complete
                 if progress >= 1.0:
                     if len(accumulated_encodings) > 3: # Require at least 3 good samples
                         # COMPUTE AVERAGE ENCODING (Centroid)
                         mean_encoding = np.mean(accumulated_encodings, axis=0)
                         
                         # Register
                         manager.id_manager.register_trusted_identity("COMMANDER", [mean_encoding])
                         print(f"[SYSTEM] Registered COMMANDER with {len(accumulated_encodings)} samples.")
                         # Hold the "Complete" state for a moment before resetting?
                         # For now, we rely on the loop continuing as long as R is held.
                     else:
                         # Failed to get enough samples
                         cv2.putText(frame, "KEEP FACE STEADY", (W//2 - 100, H//2 + 50), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                # Reset
                reg_start_time = None
                accumulated_encodings = []
    
            # --- RENDER LAYERS ---
            draw_hud(frame, turret, targets, primary, aim_mode, manager)
    
            # --- TECH DEMO VISUALS ---
            from src.visualization import draw_skeleton, draw_mediapipe_mesh
            
            # 1. Pose Skeleton (For ALL tracked persons)
            if results and results[0].keypoints is not None:
                 for i, kps in enumerate(results[0].keypoints):
                     # Get raw xy
                     kp_xy = kps.xy[0].cpu().numpy()
                     draw_skeleton(frame, kp_xy)
    
            # 2. MediaPipe Face Landmarker (Tasks API)
            # Process for ALL faces (not just primary)
            rgb_frame = cv2.cvtColor(clean_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
             
            # Process Async (Video Mode)
            detection_result = landmarker.detect_for_video(mp_image, ts_ms)
             
            if detection_result.face_landmarks:
                 draw_mediapipe_mesh(frame, detection_result.face_landmarks)
    
            # --- 4. AUTO-SNAPSHOT (After Rendering) ---
            # User requirement: "Save full frame with labels when new person enters"
            # We need to detect if a NEW person entered this frame.
            new_ids_this_frame = []
            if results:
                 run_dir = results[0].save_dir
                 if run_dir:
                     import os
                     for t in targets:
                         tid = t['id']
                         
                         # LOGIC CHANGE: Track ALL IDs, even temporary "Trk-" ones.
                         # If a new track appears (Trk-12), we snapshot it.
                         # If it later becomes "ID-05", we might snapshot again (which is fine/good).
                         # CONFIDENCE FILTER: Only snapshot if > 60% confidence to avoid "Random" ghosts
                         conf = t.get('conf', 0.0)
                         if tid not in session_snapshots and conf > 0.6:
                             new_ids_this_frame.append(tid)
                             session_snapshots.add(tid)
                     
                     if new_ids_this_frame:
                         count = len([name for name in os.listdir(run_dir) if name.startswith("image") and name.endswith(".jpg")])
                         filename = f"image{count}.jpg"
                         path = os.path.join(run_dir, filename)
                         
                         # Ensure directory exists (YOLO should create it, but safe check)
                         if not os.path.exists(run_dir): os.makedirs(run_dir)
                         
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

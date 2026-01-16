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
from src.visualization import draw_hud

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

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Calculate MS timestamp
        ts_ms = int((time.time() - start_time_s) * 1000)

        # Track with ByteTrack for ID consistency
        # classes=[0] enforces Person only filter
        results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False, classes=[0])
        
        # --- LOGIC UPDATE ---
        targets = []
        primary = None
        
        if results:
            # Pass the FULL results object (contains keypoints) AND the frame for identity
            targets = manager.select_targets(results[0], frame, aim_mode)
            primary = manager.primary_target

        # --- TURRET PID UPDATE ---
        if primary:
            # Use the calculated AIM POINT
            target_x, target_y = primary['aim_point']
            
            # Error = Target Point - Frame Center
            error_x = target_x - (W // 2)
            error_y = target_y - (H // 2)
            turret.update(error_x, error_y)
        
        # --- RENDER ---
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
         
        # Process Async (Video Mode)
        detection_result = landmarker.detect_for_video(mp_image, ts_ms)
         
        if detection_result.face_landmarks:
             draw_mediapipe_mesh(frame, detection_result.face_landmarks)

        cv2.imshow("Safe Turret Sim", frame)
        
        # Input Handling
        key = cv2.waitKey(1)
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

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

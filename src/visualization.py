import cv2
import numpy as np
from .config import SAFE_ZONES, AIM_MODES

def draw_hud(frame, turret, targets, primary, aim_mode_idx, manager):
    H, W = frame.shape[:2]
    cx, cy = W // 2, H // 2
    
    # --- 1. Safe Zones (Visualized) ---
    for zone in SAFE_ZONES:
        x1 = int(zone[0] * W)
        y1 = int(zone[1] * H)
        x2 = int(zone[2] * W)
        y2 = int(zone[3] * H)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 50), -1) 
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.putText(frame, "SAFE ZONE", (x1+10, y1+25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # --- 2. Turret Crosshair (Dynamic Sci-Fi) ---
    import math
    import time
    
    # Get simulate aim point
    aim_x, aim_y = turret.get_current_aim_point(cx, cy)
    
    # --- CROSSHAIR SNAP LOGIC ---
    # User Request: Visually force crosshair to stick to target to fix "offset/drift" feeling
    # The true turret aim (calculated above) might be slightly off due to calibration.
    # We visually override it when locked to provide "Target Acquired" feedback.
    if primary is not None:
         # Snap to the visual aim point of the target
         aim_x, aim_y = primary['aim_point']
    
    # 2a. Draw Camera Center (Reference - Servo Arm Pivot)
    cv2.circle(frame, (cx, cy), 4, (100, 100, 100), -1)
    # Servo Arm Line (Grey Line)
    cv2.line(frame, (cx, cy), (aim_x, aim_y), (50, 50, 50), 1)
    
    # 2b. Helper for Sci-Fi Elements
    def draw_bracket(img, cx, cy, w, h, color, thickness=1):
        # Draw corners
        l = w // 4 # Corner length
        # Top Left
        cv2.line(img, (cx-w, cy-h), (cx-w+l, cy-h), color, thickness)
        cv2.line(img, (cx-w, cy-h), (cx-w, cy-h+l), color, thickness)
        # Top Right
        cv2.line(img, (cx+w, cy-h), (cx+w-l, cy-h), color, thickness)
        cv2.line(img, (cx+w, cy-h), (cx+w, cy-h+l), color, thickness)
        # Bottom Left
        cv2.line(img, (cx-w, cy+h), (cx-w+l, cy+h), color, thickness)
        cv2.line(img, (cx-w, cy+h), (cx-w, cy+h-l), color, thickness)
        # Bottom Right
        cv2.line(img, (cx+w, cy+h), (cx+w-l, cy+h), color, thickness)
        cv2.line(img, (cx+w, cy+h), (cx+w, cy+h-l), color, thickness)

    # 2c. Determine Status & Color
    if primary:
        cross_color = (0, 0, 255) # Red (Locked)
        state_radius = 25 # Tighten when locked
    else:
        cross_color = (0, 255, 255) # Cyan/Yellow (Scanning)
        state_radius = 40 # Expanded when scanning

    # 2d. Rotating Inner Ring
    t = time.time()
    angle_offset = (t * 180) % 360 # 180 deg/sec rotation
    radius = 18
    
    # Draw 3 segments
    for i in range(3):
        start_angle = angle_offset + (i * 120)
        end_angle = start_angle + 60
        # cv2.ellipse needs integer center
        cv2.ellipse(frame, (aim_x, aim_y), (radius, radius), 0, start_angle, end_angle, cross_color, 1)

    # 2e. Center Dot
    cv2.circle(frame, (aim_x, aim_y), 2, (255, 255, 255), -1)
    
    # 2f. Outer Brackets (Dynamic Size)
    draw_bracket(frame, aim_x, aim_y, state_radius, state_radius, cross_color, 2 if primary else 1)
    
    # 2g. Guide Lines
    if primary:
        # Draw lines to the actual target center to show lock vector
        tx, ty = primary['aim_point']
        cv2.line(frame, (aim_x, aim_y), (tx, ty), (0, 255, 0), 1) # Green Lock Line

    # --- 3. Targets (ALL) ---
    for t in targets:
        x1, y1, x2, y2 = t['box']
        
        # Custom Logic for visualization based on Status
        is_safe = t.get('safe', False)
        
        if is_safe:
            color = (0, 165, 255) # Orange for Safe Zone
            thick = 2
            label = f"ID:{t['id']} [SAFE]"
            cv2.putText(frame, "SAFE", (x1, y1-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        elif t['locked'] and t == primary:
            color = (0, 0, 255) # Red for Engaged
            thick = 3
            label = f"ID:{t['id']} [ENGAGED]"
            # Line to Aim Point
            cv2.line(frame, (cx, cy), t['aim_point'], color, 2)
        else:
            color = (0, 255, 255) # Yellow for Others
            thick = 1
            label = f"ID:{t['id']}"
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
        
        # Visualize Aim Point for EVERYONE
        ax, ay = t['aim_point']
        cv2.circle(frame, (ax, ay), 4, (0, 0, 255), -1)

        # DEBUG: Visualize Safe Check Point
        if 'safe_check_point' in t:
             scx, scy = t['safe_check_point']
             cv2.circle(frame, (scx, scy), 4, (255, 0, 0), -1) # Blue Dot
             
    # --- 4. System Info Panel ---
    panel_w, panel_h = 240, 280 # Increased height
    panel_x = W - panel_w - 10
    panel_y = 10
    
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (0, 255, 0), 1)
    
    sx = panel_x + 10
    sy = panel_y + 25
    line_h = 20
    
    cv2.putText(frame, "TURRET CONTROL SYS", (sx, sy), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.6, (0, 255, 0), 1)
    sy += 10
    cv2.line(frame, (sx, sy), (sx+200, sy), (50, 50, 50), 1)
    sy += 20
    
    cv2.putText(frame, f"PAN ANGLE : {turret.pan_angle:+.1f}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    sy += line_h
    cv2.putText(frame, f"TILT ANGLE: {turret.tilt_angle:+.1f}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    sy += line_h
    
    mode_str = AIM_MODES.get(aim_mode_idx, "UNKNOWN")
    cv2.putText(frame, f"AIM MODE  : {mode_str}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
    sy += line_h

    # Track Mode (Auto/Manual)
    trk_mode = "MANUAL LOCK" if manager.manual_mode else "AUTO (CLOSEST)"
    color_trk = (0, 0, 255) if manager.manual_mode else (0, 255, 0)
    cv2.putText(frame, f"TRK LOGIC : {trk_mode}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_trk, 1)
    sy += line_h

    # Status Logic
    if primary:
        status = "ENGAGED"
        color_st = (0, 0, 255)
    elif any(t.get('safe') for t in targets):
        status = "SAFE ZONE ACTV"
        color_st = (0, 165, 255)
    else:
        status = "SCANNING"
        color_st = (0, 255, 0)
        
    cv2.putText(frame, f"STATUS    : {status}", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_st, 1)
    sy += line_h
    
    # Legend
    cv2.putText(frame, "[LEGEND] Blue Dot: Safety Check", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    sy += line_h
    
    cv2.line(frame, (sx, sy), (sx+200, sy), (50, 50, 50), 1)
    sy += 15
    cv2.putText(frame, "ENGAGEMENT PANEL", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    sy += 15
    
    # List IDs
    # Sort targets by ID for stable list
    sorted_targets = sorted(targets, key=lambda x: x['id'])
    for i, t in enumerate(sorted_targets):
        if i > 4: # Limit lines
            break
            
        tid = t['id']
        dist = int(t['dist_to_center'])
        is_sel = (t == primary)
        params = ""
        
        if t.get('safe'):
            color_t = (0, 165, 255)
            params = "[SAFE]"
        elif is_sel:
            color_t = (0, 255, 0) 
            params = "[LOCK]"
        else:
            color_t = (150, 150, 150)
        
        prefix = ">" if is_sel else " "
        text = f"{prefix} {tid} {params} D:{dist}"
        cv2.putText(frame, text, (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color_t, 1)
        sy += line_h
             
    # --- 5. Controls Panel (New) ---
    panel_y = H - 140
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 130), (0, 0, 0), -1)
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + 130), (0, 255, 0), 1)
    
    sx = panel_x + 10
    sy = panel_y + 25
    cv2.putText(frame, "CONTROLS", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    sy += 20
    cv2.putText(frame, "[1/2/3] Aim: Head/Body/Leg", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    sy += 15
    cv2.putText(frame, "[M] Toggle Manual Mode", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    sy += 15
    cv2.putText(frame, "[TAB] Cycle Targets (Man)", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    sy += 15
    cv2.putText(frame, "[Q] Quit", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    sy += 15
    cv2.putText(frame, "[R] Register Face", (sx, sy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

def draw_skeleton(frame, keypoints):
    """
    Draw Pose Skeleton from YOLO keypoints.
    keypoints: (17, 2) or (17, 3) numpy array
    """
    from .skeleton_constants import SKELETON_CONNECTIONS
    
    # Draw Connections
    for p1_idx, p2_idx in SKELETON_CONNECTIONS:
        if p1_idx < len(keypoints) and p2_idx < len(keypoints):
            x1, y1 = keypoints[p1_idx][:2]
            x2, y2 = keypoints[p2_idx][:2]
            
            # Check confidence if available (usually 3rd val or checks > 0)
            if x1 > 0 and x2 > 0 and y1 > 0 and y2 > 0:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)

    # Draw Joints
    for i, kp in enumerate(keypoints):
        x, y = kp[:2]
        if x > 0 and y > 0:
            color = (0, 255, 255) if i < 5 else (0, 0, 255) # Head parts yellow, body red
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

def draw_face_landmarks(frame, landmarks):
    """
    Draw 68-point face landmarks (dlib style).
    landmarks: list of (x,y) tuples or dictionary from face_recognition
    """
    # If using face_recognition dict structure
    if isinstance(landmarks, dict):
        for feature, points in landmarks.items():
            # Draw lines connecting feature points
            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], False, (255, 255, 255), 1)
            
            # Draw dots
            for (x, y) in points:
                cv2.circle(frame, (x, y), 1, (100, 255, 100), -1)

def draw_mediapipe_mesh(frame, face_landmarks_list):
    """
    Draw 468-point Face Mesh using Manual OpenCV (Robust to missing mp.solutions).
    face_landmarks_list: List of list of NormalizedLandmark objects
    """
    H, W = frame.shape[:2]
    
    for landmarks in face_landmarks_list:
        # Draw all points
        for lm in landmarks:
            cx, cy = int(lm.x * W), int(lm.y * H)
            cv2.circle(frame, (cx, cy), 1, (255, 255, 255), -1, cv2.LINE_AA)
            
        # Draw specific Contours (Approximation since we don't have connection map)
        # Lips: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
        # Left Eye: 33, 7, 163, 144, 145, 153, 154, 155, 133
        # Right Eye: 263, 249, 390, 373, 374, 380, 381, 382, 362
        
        # Simple highlight of eyes (rough indices for iris center)
        # Left Iris: 468, Right Iris: 473 (if refined)
        if len(landmarks) > 468:
            l_iris = landmarks[468]
            r_iris = landmarks[473]
            cv2.circle(frame, (int(l_iris.x * W), int(l_iris.y * H)), 3, (0, 0, 255), -1)
            cv2.circle(frame, (int(r_iris.x * W), int(r_iris.y * H)), 3, (0, 0, 255), -1)

def draw_registration_ui(frame, progress):
    """
    Overlay for Registration Mode.
    progress: 0.0 to 1.0
    """
    H, W = frame.shape[:2]
    
    # Darken background slightly
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, H), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
    
    cx, cy = W // 2, H // 2
    
    # Draw Scanning Box (Center)
    box_s = 300
    x1, y1 = cx - box_s//2, cy - box_s//2
    x2, y2 = cx + box_s//2, cy + box_s//2
    
    color = (255, 255, 255)
    if progress >= 1.0:
        color = (0, 255, 0)
        
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Progress Bar
    bar_w = 400
    bar_h = 20
    bx, by = cx - bar_w//2, y2 + 40
    
    cv2.rectangle(frame, (bx, by), (bx + bar_w, by + bar_h), (100, 100, 100), -1)
    fill_w = int(bar_w * progress)
    cv2.rectangle(frame, (bx, by), (bx + fill_w, by + bar_h), (0, 255, 255), -1)
    
    # Text
    if progress < 1.0:
        text = f"SCANNING FACE... {int(progress*100)}%"
        sub = "Turn head slightly left/right"
    else:
        text = "REGISTRATION COMPLETE"
        sub = "Identity: COMMANDER saved."
        
    cv2.putText(frame, text, (bx, by - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.putText(frame, sub, (bx, by + bar_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

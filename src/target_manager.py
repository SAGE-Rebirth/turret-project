import numpy as np
from .config import SAFE_ZONES
from .identity_manager import IdentityManager

class TargetManager:
    """
    Handles Multi-Target logic, Locking, and Safety Zones.
    Key Change: Uses IdentityManager to map transient YOLO IDs to Persistent PIDs.
    """
    def __init__(self, frame_width, frame_height):
        self.W = frame_width
        self.H = frame_height
        self.cx = frame_width // 2
        self.cy = frame_height // 2
        
        self.locked_ids = set()
        self.primary_target = None
        
        # Identity System
        self.id_manager = IdentityManager()
        
        # Manual Override
        self.manual_mode = False
        self.selected_id = None
        
    def is_safe(self, box):
        """
        Check if target is in a Safe Zone (Ignore Zone).
        box: (x1, y1, x2, y2)
        """
        # Calculate box center normalized
        # Make sure we use float division
        bx_cx = ((box[0] + box[2]) / 2.0) / float(self.W)
        bx_cy = ((box[1] + box[3]) / 2.0) / float(self.H)
        
        for zone in SAFE_ZONES:
            zx1, zy1, zx2, zy2 = zone
            # If center of target is inside the safe rect
            if zx1 <= bx_cx <= zx2 and zy1 <= bx_cy <= zy2:
                # DEBUG PRINT (Optional, remove if too spammy)
                # print(f"SAFE ZONE TRIGGER: {bx_cx:.2f}, {bx_cy:.2f} in {zone}") 
                return True # In safe zone
        return False

    def select_targets(self, results, frame, aim_mode):
        """
        Process tracker output, filter unsafe, and select Primary.
        Accepts full YOLO Results object to access Keypoints.
        Updated: Pass 'frame' for facial recognition.
        """
        valid_targets = []
        
        boxes = results.boxes
        keypoints = results.keypoints
        
        if boxes is None or boxes.id is None:
            self.primary_target = None
            return []

        # --- 1. Filter & Parse ---
        for i, box in enumerate(boxes):
            # Check Class (Strict Person Only)
            cls_id = int(box.cls[0])
            if cls_id != 0: 
                continue 
            
            # Check Safety
            xyxy = box.xyxy[0].cpu().numpy()
            is_safe_zone = self.is_safe(xyxy)
            # if is_safe_zone: continue  <-- OLD (Removed)
            
            # Extract Transient ID
            yolo_id = int(box.id[0]) if box.id is not None else -1
            
            # --- IDENTITY RESOLUTION ---
            # Map YOLO ID -> Persistent PID
            pid = self.id_manager.get_pid(frame, xyxy, yolo_id)
            
            # --- PRECISE TARGETING LOGIC (Keypoints) ---
            x1, y1, x2, y2 = map(int, xyxy)
            aim_x, aim_y = (x1 + x2) // 2, (y1 + y2) // 2 # Default to center
            
            if keypoints is not None and len(keypoints) > i:
                # Keypoints for this person
                kps = keypoints[i].xy[0].cpu().numpy() # Shape (17, 2)
                # Kps: 0=Nose, 5=LSh, 6=RSh, 11=LHip, 12=RHip, 13=LKnee, 14=RKnee
                
                # Check confidence of relevant keypoints (simplified: if not [0,0])
                def is_valid(kp_idx):
                    return kps.shape[0] > kp_idx and kps[kp_idx][0] != 0 and kps[kp_idx][1] != 0

                if aim_mode == 1: # HEAD (Precise Forehead)
                    # Use Eyes (1, 2) and Nose (0) to vector to Forehead
                    if is_valid(1) and is_valid(2): # Both Eyes
                        mid_x = (kps[1][0] + kps[2][0]) / 2
                        mid_y = (kps[1][1] + kps[2][1]) / 2
                        
                        if is_valid(0): # Nose available for vector
                            # Vector from Nose to EyeMid
                            vec_x = mid_x - kps[0][0]
                            vec_y = mid_y - kps[0][1]
                            
                            # Forehead is roughly same distance above eyes
                            # Scaling factor 1.2 to hit forehead center
                            aim_x = mid_x + (vec_x * 1.2)
                            aim_y = mid_y + (vec_y * 1.2)
                        else:
                            # Use eye distance as scale
                            eye_dist = np.sqrt((kps[1][0] - kps[2][0])**2 + (kps[1][1] - kps[2][1])**2)
                            aim_x = mid_x
                            # Move up (negative Y) by approx 0.8 * eye_dist
                            aim_y = mid_y - (eye_dist * 0.8)
                            
                    elif is_valid(0): # Only Nose
                        # Go up from nose by approx 1/6 of face height (estimated from box)
                        h = y2 - y1
                        aim_x = kps[0][0]
                        aim_y = kps[0][1] - (h * 0.15) # Approx forehead from nose
                    else:
                        # Fallback Box
                        h = y2 - y1
                        aim_y = y1 + (h * 0.08) # Top 8%

                elif aim_mode == 3: # NON_LETHAL (Legs)
                    if is_valid(13) and is_valid(14): # Knees Midpoint
                        aim_x = (kps[13][0] + kps[14][0]) / 2
                        aim_y = (kps[13][1] + kps[14][1]) / 2
                    elif is_valid(11) and is_valid(12): # Hips (aim lower than hips)
                        mid_x = (kps[11][0] + kps[12][0]) / 2
                        mid_y = (kps[11][1] + kps[12][1]) / 2
                        aim_x = mid_x
                        aim_y = mid_y + (y2 - mid_y) * 0.5 # Halfway from hips to feet
                    else:
                        # Fallback
                        h = y2 - y1
                        aim_y = y1 + (h * 0.75)

                else: # UPPER_BODY
                    if is_valid(5) and is_valid(6): # Shoulders Midpoint
                        aim_x = (kps[5][0] + kps[6][0]) / 2
                        aim_y = (kps[5][1] + kps[6][1]) / 2
                        # Adjust slightly down for chest center
                        aim_y += (y2 - y1) * 0.05 
                    else:
                        # Fallback
                        h = y2 - y1
                        aim_y = y1 + (h * 0.35)

            # Store Data
            target_data = {
                'id': pid,      # USE GLOBAL PID instead of transient ID
                'yolo_id': yolo_id, # Keep track of transient ID for debug
                'box': (x1, y1, x2, y2),
                'center': ((x1+x2)//2, (y1+y2)//2), # Box Center
                'keypoints': kps, # Store raw keypoints for advanced vis
                'aim_point': (int(aim_x), int(aim_y)), # Precise Aim Point
                'dist_to_center': np.sqrt(((x1+x2)//2 - self.cx)**2 + ((y1+y2)//2 - self.cy)**2),
                'safe_check_point': ((x1+x2)//2, (y1+y2)//2), # The point used for IsSafe check
                'safe': is_safe_zone,
                'locked': False
            }
            valid_targets.append(target_data)

        # --- 2. Select Primary ---
        best_t_data = None
        
        if self.manual_mode and self.selected_id is not None:
            # Find the selected ID (Check against PID)
            for t in valid_targets:
                if t['id'] == self.selected_id:
                    # Logic: If manual, do we allow locking on safe zone? 
                    # Let's say NO. If strictly person in safe zone, they are ignored.
                    if not t['safe']: 
                        best_t_data = t
                    break
        else:
            # Auto Mode: Closest to Center
            best_dist = float('inf')
            for t in valid_targets:
                # Ignore Safe targets for selection
                if t['safe']: continue 
                
                if t['dist_to_center'] < best_dist:
                    best_dist = t['dist_to_center']
                    best_t_data = t
        
        # --- 3. Update State ---
        if best_t_data:
            self.primary_target = best_t_data
            self.locked_ids.add(best_t_data['id'])
            
            # If in manual mode, determine selection
            if self.manual_mode and self.selected_id is None:
                self.selected_id = best_t_data['id']
        else:
            self.primary_target = None

        # Sync 'locked' status
        for ft in valid_targets:
            if ft['id'] in self.locked_ids:
                ft['locked'] = True
                
        return valid_targets

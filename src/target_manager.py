import numpy as np
from .config import SAFE_ZONES
from .identity_manager import IdentityManager

# Min YOLO keypoint confidence to treat a joint as valid for targeting.
KP_CONF_THRESHOLD = 0.5


class TargetManager:
    """
    Handles Multi-Target logic, Locking, and Safety Zones.

    Locking is anchored on the transient YOLO track ID (stable within a
    track) rather than the PID (which is async-resolved and changes from
    "Trk-N" -> "ID-NN" mid-track). PIDs are still surfaced for display.
    """
    def __init__(self, frame_width, frame_height):
        self.W = frame_width
        self.H = frame_height
        self.cx = frame_width // 2
        self.cy = frame_height // 2

        self.primary_target = None
        self.last_primary_yolo_id = None
        self.target_lost = False  # True for one frame when lock drops

        self.id_manager = IdentityManager()

        # Manual Override — anchored on YOLO track ID (transient but stable
        # within a track), not on PID (which is async-resolved).
        self.manual_mode = False
        self.selected_yolo_id = None

    def is_safe(self, box):
        bx_cx = ((box[0] + box[2]) / 2.0) / float(self.W)
        bx_cy = ((box[1] + box[3]) / 2.0) / float(self.H)
        for zx1, zy1, zx2, zy2 in SAFE_ZONES:
            if zx1 <= bx_cx <= zx2 and zy1 <= bx_cy <= zy2:
                return True
        return False

    def select_targets(self, results, frame, aim_mode):
        """
        Process tracker output, filter unsafe, and select Primary.
        Accepts full YOLO Results object to access Keypoints.
        """
        valid_targets = []
        boxes = results.boxes
        keypoints = results.keypoints

        if boxes is None or boxes.id is None:
            self._update_primary(None)
            return []

        # Prefer the full keypoint tensor (includes confidence). YOLO pose
        # exposes .data as (N, 17, 3) where the third channel is conf.
        kp_data = None
        if keypoints is not None:
            try:
                kp_data = keypoints.data.cpu().numpy()  # (N, 17, 3)
            except Exception:
                kp_data = None

        for i, box in enumerate(boxes):
            if int(box.cls[0]) != 0:
                continue

            xyxy = box.xyxy[0].cpu().numpy()
            is_safe_zone = self.is_safe(xyxy)
            yolo_id = int(box.id[0])
            pid = self.id_manager.get_pid(frame, xyxy, yolo_id)

            x1, y1, x2, y2 = map(int, xyxy)
            aim_x, aim_y = (x1 + x2) // 2, (y1 + y2) // 2

            kps = None
            if kp_data is not None and i < kp_data.shape[0]:
                kps = kp_data[i]  # (17, 3) -> (x, y, conf)

                def is_valid(idx):
                    if kps is None or idx >= kps.shape[0]:
                        return False
                    # Use confidence channel when present; fall back to (x,y)>0.
                    if kps.shape[1] >= 3:
                        return kps[idx, 2] >= KP_CONF_THRESHOLD
                    return kps[idx, 0] > 0 and kps[idx, 1] > 0

                aim_x, aim_y = self._compute_aim_point(
                    kps, is_valid, aim_mode, x1, y1, x2, y2, aim_x, aim_y
                )

            target_data = {
                'id': pid,
                'yolo_id': yolo_id,
                'conf': float(box.conf[0]),
                'box': (x1, y1, x2, y2),
                'center': ((x1 + x2) // 2, (y1 + y2) // 2),
                'keypoints': kps[:, :2] if kps is not None else None,
                'aim_point': (int(aim_x), int(aim_y)),
                'dist_to_center': np.sqrt(((x1 + x2) // 2 - self.cx) ** 2 +
                                          ((y1 + y2) // 2 - self.cy) ** 2),
                'safe_check_point': ((x1 + x2) // 2, (y1 + y2) // 2),
                'safe': is_safe_zone,
                'locked': False,
            }
            valid_targets.append(target_data)

        # --- Select Primary ---
        best = None
        if self.manual_mode and self.selected_yolo_id is not None:
            for t in valid_targets:
                if t['yolo_id'] == self.selected_yolo_id and not t['safe']:
                    best = t
                    break
        else:
            best_dist = float('inf')
            for t in valid_targets:
                if t['safe']:
                    continue
                if t['dist_to_center'] < best_dist:
                    best_dist = t['dist_to_center']
                    best = t

        # If auto mode just picked someone, remember them so a switch to
        # manual immediately locks on the right track.
        if best and not self.manual_mode:
            self.selected_yolo_id = best['yolo_id']

        self._update_primary(best)
        if best:
            best['locked'] = True
        return valid_targets

    def _compute_aim_point(self, kps, is_valid, aim_mode, x1, y1, x2, y2,
                           default_x, default_y):
        """Per-mode aim point from keypoints, with fallbacks."""
        h = y2 - y1
        aim_x, aim_y = default_x, default_y

        if aim_mode == 1:  # HEAD
            if is_valid(1) and is_valid(2):
                mid_x = (kps[1][0] + kps[2][0]) / 2
                mid_y = (kps[1][1] + kps[2][1]) / 2
                if is_valid(0):
                    vec_x = mid_x - kps[0][0]
                    vec_y = mid_y - kps[0][1]
                    # Eyes must be above nose for forehead extrapolation —
                    # otherwise (head tilted back) fall through to eye-distance.
                    if vec_y < 0:
                        aim_x = mid_x + (vec_x * 1.2)
                        aim_y = mid_y + (vec_y * 1.2)
                        return int(aim_x), int(aim_y)
                eye_dist = np.hypot(kps[1][0] - kps[2][0], kps[1][1] - kps[2][1])
                aim_x = mid_x
                aim_y = mid_y - (eye_dist * 0.8)
            elif is_valid(0):
                aim_x = kps[0][0]
                aim_y = kps[0][1] - (h * 0.15)
            else:
                aim_y = y1 + (h * 0.08)

        elif aim_mode == 3:  # NON_LETHAL
            if is_valid(13) and is_valid(14):
                aim_x = (kps[13][0] + kps[14][0]) / 2
                aim_y = (kps[13][1] + kps[14][1]) / 2
            elif is_valid(11) and is_valid(12):
                mid_x = (kps[11][0] + kps[12][0]) / 2
                mid_y = (kps[11][1] + kps[12][1]) / 2
                aim_x = mid_x
                aim_y = mid_y + (y2 - mid_y) * 0.5
            else:
                aim_y = y1 + (h * 0.75)

        else:  # UPPER_BODY
            if is_valid(5) and is_valid(6):
                aim_x = (kps[5][0] + kps[6][0]) / 2
                aim_y = (kps[5][1] + kps[6][1]) / 2
                aim_y += h * 0.05
            else:
                aim_y = y1 + (h * 0.35)

        return int(aim_x), int(aim_y)

    def _update_primary(self, new_primary):
        """Track lock transitions so callers can reset the PID controller."""
        prev = self.last_primary_yolo_id
        new_yolo = new_primary['yolo_id'] if new_primary else None
        self.target_lost = (prev is not None and new_yolo != prev)
        self.primary_target = new_primary
        self.last_primary_yolo_id = new_yolo

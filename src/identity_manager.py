import face_recognition
import numpy as np
import cv2
import time
from concurrent.futures import ThreadPoolExecutor

class IdentityManager:
    """
    Manages persistent identities using facial recognition with Asynchronous Processing.
    Ensures the main thread is NEVER blocked by face_recognition.
    """
    def __init__(self, match_tolerance=0.6, trusted_tolerance=0.5):
        # General-gallery threshold: face_recognition's standard is 0.6;
        # we use it for the auto-discovered ID-NN gallery.
        self.match_tolerance = match_tolerance
        # Trusted/VIP threshold is stricter — a false COMMANDER match has
        # higher consequence than a false ID-NN match.
        self.trusted_tolerance = trusted_tolerance
        
        # Database: { 'PID-1': {'encodings': [np.array(...)], 'last_seen': ts} }
        self.known_entities = {}
        
        # Trusted Identities: { 'COMMANDER': [encodings...] }
        self.trusted_identities = {}
        
        self.next_pid_counter = 1
        
        # Session Mapping: { yolo_id: 'PID-1' }
        self.yolo_to_pid = {}
        
        # Check Limiter: { yolo_id: last_check_time }
        self.last_check_time = {}
        self.check_interval = 1.0 # Check every 1s per ID
        # Once a yolo_id has matched a stable PID this many times, stop re-checking.
        self.confirm_threshold = 3
        self.confirm_counts = {} # { yolo_id: int }
        self.confirmed_ids = set() # yolo_ids whose PID is locked in

        # Async Executor
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.pending_tasks = set() # Track YOLO IDs currently being processed
        # Hard cap on inflight identity work. With max_workers=1 the queue is
        # effectively the pending_tasks set; cap it to avoid backpressure when
        # many tracks appear at once.
        self.max_inflight = 4
        
        print(f"[SYSTEM] AsyncIdentityManager Initialized (Tol={self.match_tolerance})")
        print("[SYSTEM] Deep Metric Learning Model: ResNet-34 (dlib) Active")

    def encode_async(self, bgr_crop):
        """Submit a face crop for encoding. Returns a Future yielding the
        first encoding (np.ndarray) or None if no face was found.

        Off-loads face_recognition.face_encodings from the main thread so the
        registration loop doesn't block the display."""
        return self.executor.submit(self._encode_crop_bg, bgr_crop)

    def _encode_crop_bg(self, bgr_crop):
        try:
            h, w = bgr_crop.shape[:2]
            if w > 200:
                scale = 200 / w
                bgr_crop = cv2.resize(bgr_crop, (0, 0), fx=scale, fy=scale)
            rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
            encs = face_recognition.face_encodings(rgb)
            return encs[0] if len(encs) > 0 else None
        except Exception as e:
            print(f"[ERROR] encode_async: {e}")
            return None

    def register_trusted_identity(self, name, encodings):
        """
        Explicitly register a VIP/Trusted Identity.
        encodings: List of face encodings (typically a single centroid).

        Appends to existing samples for this name rather than replacing,
        so re-running registration augments the gallery instead of
        clobbering it. Cap at 10 to bound match cost.
        """
        existing = self.trusted_identities.get(name, [])
        existing.extend(encodings)
        self.trusted_identities[name] = existing[-10:]

        # Invalidate confirmation cache: tracks that were "confirmed" as
        # ID-NN before this registration must get another chance to match
        # the trusted identity, otherwise they stay frozen on the old PID.
        self.confirmed_ids.clear()
        self.confirm_counts.clear()
        self.last_check_time.clear()

        print(f"[IDENTITY] Registered Trusted Identity: {name} "
              f"({len(self.trusted_identities[name])} total samples)")

    def get_pid(self, frame, box, yolo_id):
        """
        Non-blocking PID resolution.
        Returns:
            - Existing PID if known.
            - "Scanning..." if currently processing.
            - "Trk-ID" if unknown and waiting for slot.
        """
        # 1. Return cached result if available
        if yolo_id in self.yolo_to_pid:
            # Confirmed PIDs never re-scan — saves a face_recognition call
            # per active track per second for the rest of the session.
            if yolo_id in self.confirmed_ids:
                return self.yolo_to_pid[yolo_id]
            last_time = self.last_check_time.get(yolo_id, 0)
            if time.time() - last_time < self.check_interval:
                return self.yolo_to_pid[yolo_id]
            # Else: Proceed to schedule a re-check (fall through)

        # 2. Check if already being processed
        if yolo_id in self.pending_tasks:
            # Return current best guess (or Scanning)
            return self.yolo_to_pid.get(yolo_id, f"Scanning...")

        # 3. Schedule Background Task
        # Copy necessary data (Frame crop) to avoid race conditions with frame buffer
        x1, y1, x2, y2 = map(int, box)
        H, W = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        
        # Skip invalid boxes
        if (x2 - x1) < 20 or (y2 - y1) < 20:
             return self.yolo_to_pid.get(yolo_id, f"Trk-{yolo_id}")

        # Drop new identity work when the executor is saturated. The track
        # will be re-checked on a later frame.
        if len(self.pending_tasks) >= self.max_inflight:
             return self.yolo_to_pid.get(yolo_id, f"Trk-{yolo_id}")

        # Crop the FACE region (top ~40% of bbox) instead of the full body
        # so the encoding compares cleanly to registration samples (which
        # also use the face region). face_recognition's HOG detector
        # struggles when the face is a small fraction of the crop.
        bh = y2 - y1
        face_y2 = min(y2, y1 + max(int(bh * 0.4), 60))
        crop = frame[y1:face_y2, x1:x2].copy()  # COPY is crucial for threads

        self.pending_tasks.add(yolo_id)
        self.executor.submit(self._process_face_bg, crop, yolo_id)
        
        # Return immediate fallback
        return self.yolo_to_pid.get(yolo_id, f"Scanning...")

    def _process_face_bg(self, crop, yolo_id):
        """
        Background Worker Function.
        """
        try:
            # Optimization: Resize large faces to standard size (e.g. 200px width)
            # Face recognition is O(N^2) with pixels, resizing speeds up 10x
            h, w = crop.shape[:2]
            if w > 200:
                scale = 200 / w
                crop = cv2.resize(crop, (0,0), fx=scale, fy=scale)

            # Convert to RGB
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Detect
            encodings = face_recognition.face_encodings(rgb_crop)
            
            if len(encodings) == 0:
                # No face found
                self.yolo_to_pid[yolo_id] = f"Trk-{yolo_id}"
            else:
                current_encoding = encodings[0]
                found_pid, dist = self._match_encoding(current_encoding)
                
                if not found_pid:
                    # New Identity
                    found_pid = f"ID-{self.next_pid_counter:02d}"
                    self.next_pid_counter += 1
                    self.known_entities[found_pid] = {
                        'encodings': [current_encoding], # Start Gallery
                        'created_at': time.time(),
                        'last_seen': time.time()
                    }
                    print(f"[IDENTITY] New Persistent ID: {found_pid} (No match found)")
                else:
                    # Found existing match (Trusted or Known)
                    if found_pid in self.known_entities:
                        self.known_entities[found_pid]['last_seen'] = time.time()
                        
                        # Add to gallery if match is good but not identical (Learning)
                        # Limit gallery size to 5
                        if len(self.known_entities[found_pid]['encodings']) < 5:
                            self.known_entities[found_pid]['encodings'].append(current_encoding)

                    print(f"[IDENTITY] Matched {found_pid} (Dist: {dist:.3f})")

                # Update Mapping
                prior = self.yolo_to_pid.get(yolo_id)
                self.yolo_to_pid[yolo_id] = found_pid

                # Promote to "confirmed" after N consecutive matches to the same PID.
                if prior == found_pid:
                    self.confirm_counts[yolo_id] = self.confirm_counts.get(yolo_id, 1) + 1
                    if self.confirm_counts[yolo_id] >= self.confirm_threshold:
                        self.confirmed_ids.add(yolo_id)
                else:
                    self.confirm_counts[yolo_id] = 1

        except Exception as e:
            print(f"[ERROR] bg_process: {e}")
        finally:
            if yolo_id in self.pending_tasks:
                self.pending_tasks.remove(yolo_id)
            self.last_check_time[yolo_id] = time.time()

    def _match_encoding(self, encoding):
        """
        Helper: Compare encoding against database (Gallery Match).
        Returns (pid, distance) or (None, None)
        """
        best_dist = 1.0
        found_pid = None
        
        # 1. Check Trusted/VIP Identities FIRST with stricter threshold —
        # a false COMMANDER match is more costly than a false ID-NN.
        for name, encodings in self.trusted_identities.items():
            distances = face_recognition.face_distance(encodings, encoding)
            min_dist = min(distances) if len(distances) > 0 else 1.0

            if min_dist < self.trusted_tolerance and min_dist < best_dist:
                best_dist = min_dist
                found_pid = name # e.g. "COMMANDER"

        # If trusted found, return immediately? Or define policy.
        # Let's return trusted immediately if found, as it takes precedence.
        if found_pid:
            return found_pid, best_dist

        # 2. Check General Known Identities
        # Improve locking: Iterate dictionary safely
        known_items = list(self.known_entities.items())
        
        for pid, data in known_items:
            # Check against ALL stored encodings for this person
            # Returns list of distances
            distances = face_recognition.face_distance(data['encodings'], encoding)
            
            # Use the MINIMUM distance (best match in gallery)
            min_dist = min(distances) if len(distances) > 0 else 1.0
            
            if min_dist < self.match_tolerance and min_dist < best_dist:
                best_dist = min_dist
                found_pid = pid
                
        return found_pid, best_dist

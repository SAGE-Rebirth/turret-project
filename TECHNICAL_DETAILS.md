# Technical Details

## Tracking Logic
The system uses `ultralytics` YOLO models (specifically `yolo11n-pose.pt` by default) to detect people and extract skeletal keypoints.
- **ByteTrack** is used for ID persistence across frames.
- **Pose Estimation** allows for sub-object precision. Instead of aiming at the bounding box center, we calculate specific vectors based on eyes/nose (Head mode), shoulders (Upper Body mode), or knees/hips (Non-lethal mode).

## Turret Controller (PID)
The `TurretController` class implements a Proportional-Integral-Derivative (PID) controller for both Pan (X) and Tilt (Y) axes.

- **Inputs**: Pixel error (difference between target point and screen center).
- **Outputs**: Servo angle adjustments (simulated).
- **Limits**:
  - Max speed (degrees per frame).
  - Mechanical angle limits (e.g. -90 to 90 degrees).

## Coordinate Systems
- **World/Frame**: Top-left is (0,0).
- **Safety Zones**: Defined in normalized coordinates (0.0 to 1.0) to be resolution independent.
  - `(0.0, 0.0, 0.15, 1.0)` defines a 15% strip on the left edge.
- **Angles**:
  - Center is 0, 0.
  - Pan: Positive = Right, Negative = Left.
  - Tilt: Positive = Up, Negative = Down (Inverted from pixel Y-axis).

## Facial Tracking (The "Iron Man" HUD)
The system now uses **Google MediaPipe Face Landmarker** (Tasks API) for high-fidelity facial visualization.
- **Mesh**: 478 3D landmarks covering the entire face (tesselation), iris, lips, and eyebrows.
- **Tasks API**: Uses the modern `vision.FaceLandmarker` which is more robust than the legacy `solutions` API.
- **Data Flow**:
    1.  Frame captured.
    2.  YOLO detects Person & Pose.
    3.  If a target is LOCKED, the frame is sent to MediaPipe.
    4.  MediaPipe returns dense landmarks for UI rendering.

## Dependencies
- **OpenCV**: Image capture and HUD drawing.
- **NumPy**: Vector math and PID calculations.
- **Ultralytics**: YOLO model inference and tracking.
- **MediaPipe**: Dense Face Mesh and Landmark detection.
- **dlib & face_recognition**: Identity persistence (ResNet-34).

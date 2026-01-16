# Troubleshooting & Common Errors

## MediaPipe Issues

### 1. `AttributeError: module 'mediapipe' has no attribute 'solutions'`
**Cause**: 
- Newer versions of `mediapipe` (0.10.x+) often ship with the new **Tasks API** and might have the legacy `solutions` API separated or not lazy-loaded correctly on some platforms (especially macOS ARM64).
- It can also be caused by a local file named `mediapipe.py` shadowing the library.

**Fix**:
- **Check for conflicts**: Ensure you don't have a file named `mediapipe.py` in your project folder.
- **Use Tasks API**: This project has been updated to use the new `mediapipe.tasks` API, which avoids this error entirely.
- **Explicit Import**: If you must use solutions, try:
  ```python
  import mediapipe.python.solutions.face_mesh
  ```

### 2. `ModuleNotFoundError: No module named 'mediapipe.python'`
**Cause**:
- Inconsistent package structure across OS/Pip versions. Some installs put `solutions` directly under `mediapipe`, others under `mediapipe.python`.

**Fix**:
- Reinstall the package: `pip install --force-reinstall mediapipe`
- Use the **Tasks API** (Recommended).

## Models

### 1. Missing `face_landmarker.task`
**Error**: `FileNotFoundError: ... face_landmarker.task`
**Fix**:
- You must download the official model bundle from Google.
- Command:
  ```bash
  curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
  ```

## dlib Issues (macOS)

### 1. Installation Fails (CMake/Boost)
**Cause**: `dlib` requires a C++ compiler and CMake.
**Fix**:
- Install dependencies:
  ```bash
  brew install cmake
  brew install boost
  pip install dlib
  ```

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

## dlib Issues (Windows)

### 1. `RuntimeError: CMake must be installed to build the following extensions: dlib`
**Cause**: 
`dlib` is a C++ library. On Windows, PyPI often does not provide pre-compiled binary "wheels" for your specific Python version. Therefore, `pip` tries to download the source code and **compile it locally** on your machine. This requires:
1.  **CMake**: To generate the build files.
2.  **C++ Compiler**: To actually compile the code (Visual Studio Build Tools).

**Fix (Step-by-Step)**:

**Step A: Install CMake**
1.  Download the **Windows x64 Installer** from [cmake.org/download](https://cmake.org/download/).
2.  Run the installer.
3.  **CRITICAL**: Check the box **"Add CMake to the system PATH for all users"** during installation.
4.  Restart your computer (or at least your terminal) to ensure the PATH is updated.
5.  Verify by running `cmake --version` in your terminal.

**Step B: Install Visual Studio Build Tools**
If it still fails with "Visual Studio not found":
1.  Download **Visual Studio Build Tools** from [visualstudio.microsoft.com/downloads](https://visualstudio.microsoft.com/downloads/).
2.  Run the installer.
3.  Select the **"Desktop development with C++"** workload.
4.  Make sure **"MSVC ... C++ x64/x86 build tools"** is checked on the right side.
5.  Install and wait for it to finish (it's large, ~2GB+).

**Step C: Install dlib**
Now try again:
```bash
pip install dlib
```
It should now compile successfully (this may take 5-10 minutes).

---

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

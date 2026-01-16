# 🛡️ Safe Turret System (AI-Powered)

> **A real-time, computer vision-based tracking and simulated turret control system.**  
> *Powered by YOLOv11 Pose Estimation, MediaPipe Face Mesh, and ResNet Identity Persistence.*

---

## 📖 Table of Contents
- [Features](#-features)
- [Prerequisites](#-prerequisites)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Controls](#-controls)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Technical Details](#-technical-details)
- [Contributing](#-contributing)

---

## ✨ Features
- **🎯 Precise Targeting**: Uses Skeletal Keypoints (Eyes, Shoulders, Knees) to aim at specific body parts, not just a bounding box.
- **🤖 Iron Man HUD**: Advanced facial visualization using **MediaPipe Face Mesh** (478 points) that tracks eyes, lips, and expressions in real-time.
- **🔑 Identity Persistence**: Remembers who is who (ID-01, ID-02) even if they leave and re-enter the frame, using Deep Metric Learning (standard `dlib` ResNet).
- **⚙️ PID Control**: Simulates smooth, robotic turret movement for 2-axis control.
- **🛡️ Safety Zones**: Configurable areas where targets are automatically ignored/safe.

---

## 💻 Prerequisites
- **OS**: macOS (Silicon/Intel), Windows, or Linux.
- **Python**: Version 3.10 or higher.
- **Hardware**: Webcam (Laptop or USB).

---

## 🚀 Installation & Setup

Follow these steps exactly to get the system running.

### 1. Clone the Repository
Open your terminal and clone this project:
```bash
git clone <repository-url>
cd turret-project
```

### 2. Create a Virtual Environment (Recommended)
This keeps your project dependencies clean.
```bash
python3 -m venv test-env
source test-env/bin/activate  # On Windows use: test-env\Scripts\activate
```

### 3. Install Dependencies
Install all required Python libraries (OpenCV, YOLO, MediaPipe, etc.):
```bash
pip install -r requirements.txt
```

### 4. Download AI Models (Critical!)
The system needs specific AI model files to work.
- **YOLOv11 Pose**: Automatically downloaded on first run.
- **Face Landmarker**: You **MUST** download this manually using the command below:

```bash
curl -L -o face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

---

## 🎮 Usage

Once installed, simply run the main script. Ensure your webcam is connected.

```bash
python main.py
```

Arguments:
- Currently, all settings are in `src/config.py` or default in `main.py`.

---

## ⌨️ Controls

| Key | Action | Description |
| :---: | :--- | :--- |
| **`m`** | **Toggle Mode** | Switch between **Auto-Track** (locks closest) and **Manual** (user selects). |
| **`TAB`** | **Cycle Target** | In Manual Mode, switches lock to the next person. |
| **`1`** | **Head Aim** | Turret aims at the Forehead (Lethal). |
| **`2`** | **Body Aim** | Turret aims at the Chest/Upper Body (Standard - *Default*). |
| **`3`** | **Legs Aim** | Turret aims at Knees/Legs (Non-Lethal). |
| **`q`** | **Quit** | Exit the program. |
| **`r`** | **Register Face** | (Experimental) Hold to register a "Trusted Identity". |

---

## 📂 Project Structure

Understanding the files in this project:

- **`main.py`**  
  The brain of the operation. Initializes the camera, models, and runs the main loop.
  
- **`bytetrack.yaml`**  
  Configuration for the **ByteTrack** algorithm. This ensures that "Person A" stays "Person A" as they move around.
  
- **`face_landmarker.task`**  
  The binary AI model file from Google used for the detailed face mesh. (Downloaded in Step 4).

- **`requirements.txt`**  
  List of all Python libraries needed (`ultralytics`, `mediapipe`, `opencv-python`, etc.).

- **`runs/`** (Auto-Generated)
  - Created by YOLO (Ultralytics). Contains logs, metrics, and debug images from the computer vision model. You can safely delete this if you don't need tracking history.

- **`src/`** (Source Code Folder)
  - **`turret_controller.py`**: Contains the **PID Logic** that calculates how much to turn the servo to face the target.
  - **`target_manager.py`**: Decides *who* to shoot (filtering Safe Zones, sorting by distance).
  - **`identity_manager.py`**: Handles Face Recognition (storing "known" faces).
  - **`visualization.py`**: Draws the cool "Iron Man" HUD, skeletons, and face grids.
  - **`config.py`**: Settings for Safe Zones boundaries and Servo speeds.

- **`ERROR.md`**  
  A guide to fixing common setup errors (especially for MediaPipe).

- **`TECHNICAL_DETAILS.md`**  
  Deep dive into the algorithms, coordinate systems, and math used.

---

## 🔧 Troubleshooting

If you crash or see errors like `AttributeError` or `ModuleNotFoundError`, **please read [ERROR.md](ERROR.md)**.
It contains fixes for common issues like:
- MediaPipe "solutions" attribute missing.
- Missing model files.
- macOS installation quirks.

---

## 🤝 How to Contribute

We welcome improvements! 

1.  **Fork** the repository.
2.  Create a **Branch** for your feature (`git checkout -b feature/AmazingHUD`).
3.  **Commit** your changes (`git commit -m "Added a new flashy radar"`).
4.  **Push** to the branch (`git push origin feature/AmazingHUD`).
5.  Open a **Pull Request**.

---

*Verified on macOS (M4) and Python 3.11.14*

# 🏀 Swichy — AI Basketball Trainer

Swichy is a real-time AI basketball training system that uses computer vision and pose estimation to analyze shooting form, track body mechanics, and provide intelligent coaching feedback.

Built with **Python, MediaPipe, and OpenCV**, Swichy turns raw movement into actionable basketball insights.

---

## 🚀 Features

- 📍 Real-time full-body pose detection (MediaPipe)
- 📊 Joint coordinate extraction (shoulder, elbow, wrist, knee, etc.)
- 📐 Angle analysis (elbow, knee, posture alignment)
- 🏀 Shot detection pipeline (jump → extension → release → follow-through)
- 🎯 Motion tracking over time (temporal biomechanics)
- ⚠️ Real-time coaching feedback & posture warnings
- 🎥 Live webcam / video / image processing support

---

## 🧠 System Architecture

Swichy is designed in modular layers:

### 🔹 Core Vision Engine
- Detects human pose landmarks
- Computes joint angles
- Handles drawing & visualization

### 🔹 Analysis Layer
- Angle calculations (elbow, knee, shoulder)
- Movement interpretation logic
- Pose correction rules

### 🔹 Mode System
- Live camera stream
- Image analysis
- Video processing

### 🔹 Output Layer
- Annotated images
- Processed videos
- Debug logs & metrics

---

## 📁 Project Structure

```bash
swichy/
│
├── main.py
├── pose_landmarker_full.task
│
├── config/
│   └── settings.py
│
├── core/
│   ├── detector.py
│   ├── drawing.py
│   ├── angles.py
│   ├── landmarks.py
│   └── utils.py
│
├── modes/
│   ├── live_stream.py
│   ├── image_mode.py
│   └── video_mode.py
│
├── outputs/
│   ├── images/
│   └── videos/
│
└── assets/
    ├── test.jpg
    └── test.mp4
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/swichy.git
cd swichy
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
```

### 3️⃣ Activate virtual environment

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 4️⃣ Install dependencies

```bash
pip install mediapipe opencv-python numpy
```

Or using `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## 📦 Required Libraries

- `mediapipe` → AI pose estimation
- `opencv-python` → Computer vision & webcam processing
- `numpy` → Mathematical operations

---

## ▶️ Run Swichy

```bash
python main.py
```

---

## 🛠 Example requirements.txt

```txt
mediapipe
opencv-python
numpy
```

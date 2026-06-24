# 📁 PROJECT EXPORT FOR LLMs

## 📊 Project Information

- **Project Name**: `Swishy`
- **Generated On**: 2026-06-22 05:08:42 (Asia/Damascus / GMT+03:00)
- **Total Files Processed**: 25
- **Export Tool**: Easy Whole Project to Single Text File for LLMs v1.1.0
- **Tool Author**: Jota / José Guilherme Pandolfi

### ⚙️ Export Configuration

| Setting | Value |
|---------|-------|
| Language | `en` |
| Max File Size | `1 MB` |
| Include Hidden Files | `false` |
| Output Format | `both` |

## 🌳 Project Structure

```
├── 📁 assets/
│   ├── 📄 test.jpg (5.06 KB)
│   └── 📄 Untitled.mov (26.17 MB)
├── 📁 config/
│   ├── 📁 __pycache__/
│   │   └── 📄 settings.cpython-312.pyc (381 B)
│   └── 📄 settings.py (220 B)
├── 📁 core/
│   ├── 📁 __pycache__/
│   │   ├── 📄 angles.cpython-312.pyc (1.22 KB)
│   │   ├── 📄 detector.cpython-312.pyc (2.59 KB)
│   │   ├── 📄 drawing.cpython-312.pyc (3.78 KB)
│   │   └── 📄 landmarks.cpython-312.pyc (1.41 KB)
│   ├── 📄 angles.py (741 B)
│   ├── 📄 detector.py (1.99 KB)
│   ├── 📄 drawing.py (4.83 KB)
│   ├── 📄 landmarks.py (1.52 KB)
│   └── 📄 utils.py
├── 📁 models/
│   ├── 📄 pose_landmarker_full.task (8.96 MB)
│   ├── 📄 pose_landmarker_heavy.task (29.24 MB)
│   └── 📄 pose_landmarker_lite.task (5.51 MB)
├── 📁 modes/
│   ├── 📁 __pycache__/
│   │   ├── 📄 image_mode.cpython-312.pyc (1.48 KB)
│   │   ├── 📄 live_stream.cpython-312.pyc (2.99 KB)
│   │   └── 📄 video_mode.cpython-312.pyc (2.49 KB)
│   ├── 📄 image_mode.py (904 B)
│   ├── 📄 live_stream.py (2.95 KB)
│   └── 📄 video_mode.py (2.33 KB)
├── 📄 main.ipynb (5.49 MB)
├── 📄 main.py (638 B)
└── 📄 README.md (2.75 KB)
```

## 📑 Table of Contents

**Project Files:**

- [📄 config/settings.py](#📄-config-settings-py)
- [📄 core/angles.py](#📄-core-angles-py)
- [📄 core/detector.py](#📄-core-detector-py)
- [📄 core/drawing.py](#📄-core-drawing-py)
- [📄 core/landmarks.py](#📄-core-landmarks-py)
- [📄 core/utils.py](#📄-core-utils-py)
- [📄 modes/image_mode.py](#📄-modes-image-mode-py)
- [📄 modes/live_stream.py](#📄-modes-live-stream-py)
- [📄 modes/video_mode.py](#📄-modes-video-mode-py)
- [📄 main.py](#📄-main-py)
- [📄 README.md](#📄-readme-md)

---

## 📈 Project Statistics

| Metric | Count |
|--------|-------|
| Total Files | 25 |
| Total Directories | 8 |
| Text Files | 11 |
| Binary Files | 14 |
| Total Size | 75.42 MB |

### 📄 File Types Distribution

| Extension | Count |
|-----------|-------|
| `.py` | 10 |
| `.pyc` | 8 |
| `.task` | 3 |
| `.jpg` | 1 |
| `.mov` | 1 |
| `.ipynb` | 1 |
| `.md` | 1 |

## 💻 File Code Contents

## 🚫 Binary/Excluded Files

The following files were not included in the text content:

- `assets/test.jpg`
- `assets/Untitled.mov`

## 🚫 Binary/Excluded Files

The following files were not included in the text content:

- `config/__pycache__/settings.cpython-312.pyc`

### <a id="📄-config-settings-py"></a>📄 `config/settings.py`

**File Info:**
- **Size**: 220 B
- **Extension**: `.py`
- **Language**: `python`
- **Location**: `config/settings.py`
- **Relative Path**: `config`
- **Created**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **Modified**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **MD5**: `f4d972f66b2fb554485ea3c76fdf9459`
- **SHA256**: `22f4ddd6d9ea81136a22246375b55d0e75bc15da411fe907c5815b3dc127f1e8`
- **Encoding**: ASCII

**File code content:**

```python
MODEL_PATH = "models/pose_landmarker_full.task"

WINDOW_NAME = "Basketball Pose Detection"

# Confidence thresholds
MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5

```

---

## 🚫 Binary/Excluded Files

The following files were not included in the text content:

- `core/__pycache__/angles.cpython-312.pyc`
- `core/__pycache__/detector.cpython-312.pyc`
- `core/__pycache__/drawing.cpython-312.pyc`
- `core/__pycache__/landmarks.cpython-312.pyc`

### <a id="📄-core-angles-py"></a>📄 `core/angles.py`

**File Info:**
- **Size**: 741 B
- **Extension**: `.py`
- **Language**: `python`
- **Location**: `core/angles.py`
- **Relative Path**: `core`
- **Created**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **Modified**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **MD5**: `c9aa6a702ccae2d979fb1c99f2bf2cbc`
- **SHA256**: `29963470e28ed4e26ca518adbcaa50a772e38b0eaf0a26731a919f9e91525f1f`
- **Encoding**: ASCII

**File code content:**

```python
import numpy as np


def calculate_angle(a, b, c):
    """
    Calculate angle between 3 points.

    Parameters:
    a, b, c -> tuples/lists like:
    (x, y)

    Angle is calculated at point b.

    Example:
    shoulder, elbow, wrist
    """

    # Convert to numpy arrays
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    c = np.array(c, dtype=np.float32)

    # Calculate radians
    radians = (
        np.arctan2(c[1] - b[1], c[0] - b[0]) -
        np.arctan2(a[1] - b[1], a[0] - b[0])
    )

    # Convert radians to degrees
    angle = np.abs(np.degrees(radians))

    # Keep angle between 0 and 180
    if angle > 180:
        angle = 360 - angle

    return angle

```

---

### <a id="📄-core-detector-py"></a>📄 `core/detector.py`

**File Info:**
- **Size**: 1.99 KB
- **Extension**: `.py`
- **Language**: `python`
- **Location**: `core/detector.py`
- **Relative Path**: `core`
- **Created**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **Modified**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **MD5**: `52c7aab74c7aabe806d6c273f0ffd891`
- **SHA256**: `2a62581e5e77489d772d29a4ee29d8eb225712f04a0cfbaee8ab93680af855ca`
- **Encoding**: ASCII

**File code content:**

```python
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from config.settings import (
    MODEL_PATH,
    MIN_POSE_DETECTION_CONFIDENCE,
    MIN_TRACKING_CONFIDENCE,
    MIN_PRESENCE_CONFIDENCE
)


class PoseDetector:

    def __init__(
        self,
        running_mode,
        result_callback=None
    ):

        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.PoseLandmarkerOptions(

            base_options=base_options,

            running_mode=running_mode,

            result_callback=result_callback,

            min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,

            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,

            min_pose_presence_confidence=MIN_PRESENCE_CONFIDENCE,

            num_poses=1,

            output_segmentation_masks=False
        )

        self.landmarker = vision.PoseLandmarker.create_from_options(
            options
        )

    # IMAGE MODE
    def detect_image(self, rgb_image):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        result = self.landmarker.detect(
            mp_image
        )

        return result

    # VIDEO MODE
    def detect_video_frame(
        self,
        rgb_image,
        timestamp_ms
    ):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        return result

    # LIVE STREAM MODE
    def detect_async(
        self,
        rgb_image,
        timestamp_ms
    ):

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_image
        )

        self.landmarker.detect_async(
            mp_image,
            timestamp_ms
        )

```

---

### <a id="📄-core-drawing-py"></a>📄 `core/drawing.py`

**File Info:**
- **Size**: 4.83 KB
- **Extension**: `.py`
- **Language**: `python`
- **Location**: `core/drawing.py`
- **Relative Path**: `core`
- **Created**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **Modified**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **MD5**: `93c738dc1143ddcb7156e8416e81ef18`
- **SHA256**: `c27bad871a001d358b8c005cad2826f894d38a06c64d84ea52733b0183b29e44`
- **Encoding**: ASCII

**File code content:**

```python
import cv2
import numpy as np

from core.angles import calculate_angle


# ==========================================
# MANUAL POSE CONNECTIONS
# ==========================================

POSE_CONNECTIONS = [

    # Face to shoulders
    (0, 11),
    (0, 12),

    # Shoulders
    (11, 12),

    # Left arm
    (11, 13),
    (13, 15),

    # Right arm
    (12, 14),
    (14, 16),

    # Torso
    (11, 23),
    (12, 24),
    (23, 24),

    # Left leg
    (23, 25),
    (25, 27),

    # Right leg
    (24, 26),
    (26, 28)
]


# ==========================================
# DRAW FUNCTION
# ==========================================

def draw_landmarks_on_image(
    rgb_image,
    detection_result,
    landmarks_data=None
):

    annotated_image = np.copy(rgb_image)

    if not detection_result.pose_landmarks:
        return annotated_image

    height, width, _ = annotated_image.shape

    # ==========================================
    # LOOP THROUGH DETECTED POSES
    # ==========================================

    for pose_landmarks in detection_result.pose_landmarks:

        # ==========================================
        # DRAW CONNECTIONS
        # ==========================================

        for connection in POSE_CONNECTIONS:

            start_idx, end_idx = connection

            start_landmark = pose_landmarks[start_idx]
            end_landmark = pose_landmarks[end_idx]

            start_point = (
                int(start_landmark.x * width),
                int(start_landmark.y * height)
            )

            end_point = (
                int(end_landmark.x * width),
                int(end_landmark.y * height)
            )

            cv2.line(
                annotated_image,
                start_point,
                end_point,
                (0, 255, 0),
                2
            )

        # ==========================================
        # DRAW LANDMARK POINTS + IDS
        # ==========================================

        for idx, landmark in enumerate(pose_landmarks):

            pixel_x = int(landmark.x * width)
            pixel_y = int(landmark.y * height)

            # Draw point
            cv2.circle(
                annotated_image,
                (pixel_x, pixel_y),
                5,
                (255, 0, 0),
                -1
            )

            # Draw landmark index
            cv2.putText(
                annotated_image,
                str(idx),
                (pixel_x, pixel_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1
            )

        # ==========================================
        # ELBOW ANGLE
        # ==========================================

        shoulder = pose_landmarks[12]
        elbow = pose_landmarks[14]
        wrist = pose_landmarks[16]

        shoulder_coords = [shoulder.x, shoulder.y]
        elbow_coords = [elbow.x, elbow.y]
        wrist_coords = [wrist.x, wrist.y]

        elbow_angle = calculate_angle(
            shoulder_coords,
            elbow_coords,
            wrist_coords
        )

        elbow_x = int(elbow.x * width)
        elbow_y = int(elbow.y * height)

        # Coaching feedback
        if elbow_angle < 70:

            elbow_color = (0, 0, 255)
            elbow_text = "Elbow Too Bent"

        else:

            elbow_color = (0, 255, 0)
            elbow_text = "Good Elbow"

        # Draw elbow angle
        cv2.putText(
            annotated_image,
            f"Elbow: {int(elbow_angle)}",
            (elbow_x, elbow_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            elbow_color,
            2
        )

        # Draw coaching text
        cv2.putText(
            annotated_image,
            elbow_text,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            elbow_color,
            2
        )

        # ==========================================
        # KNEE ANGLE
        # ==========================================

        hip = pose_landmarks[24]
        knee = pose_landmarks[26]
        ankle = pose_landmarks[28]

        hip_coords = [hip.x, hip.y]
        knee_coords = [knee.x, knee.y]
        ankle_coords = [ankle.x, ankle.y]

        knee_angle = calculate_angle(
            hip_coords,
            knee_coords,
            ankle_coords
        )

        knee_x = int(knee.x * width)
        knee_y = int(knee.y * height)

        cv2.putText(
            annotated_image,
            f"Knee: {int(knee_angle)}",
            (knee_x, knee_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    return annotated_image

```

---

### <a id="📄-core-landmarks-py"></a>📄 `core/landmarks.py`

**File Info:**
- **Size**: 1.52 KB
- **Extension**: `.py`
- **Language**: `python`
- **Location**: `core/landmarks.py`
- **Relative Path**: `core`
- **Created**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **Modified**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **MD5**: `9c58a75a369fc861d894b10824d9c3a9`
- **SHA256**: `e7bfb61464d3d1185ffdfa5548c9eaf91d550d3ee7cf3e8c6e0af8ae89588782`
- **Encoding**: ASCII

**File code content:**

```python
# core/landmarks.py


POSE_LANDMARKS = {

    "nose": 0,

    "left_shoulder": 11,
    "right_shoulder": 12,

    "left_elbow": 13,
    "right_elbow": 14,

    "left_wrist": 15,
    "right_wrist": 16,

    "left_hip": 23,
    "right_hip": 24,

    "left_knee": 25,
    "right_knee": 26,

    "left_ankle": 27,
    "right_ankle": 28
}


def extract_landmarks(
    detection_result,
    width,
    height
):
    """
    Extract important pose landmarks.

    Returns:
    dictionary containing:
    - pixel coordinates
    - normalized coordinates
    - depth
    - visibility
    """

    # No pose detected
    if not detection_result.pose_landmarks:
        return None

    # First detected pose
    pose = detection_result.pose_landmarks[0]

    data = {}

    for name, idx in POSE_LANDMARKS.items():

        landmark = pose[idx]

        # Normalized coordinates (0 -> 1)
        x_norm = landmark.x
        y_norm = landmark.y

        # Pixel coordinates
        x = int(x_norm * width)
        y = int(y_norm * height)

        # Depth
        z = landmark.z

        # Visibility confidence
        visibility = landmark.visibility

        data[name] = {

            # Pixel coordinates
            "x": x,
            "y": y,

            # Normalized coordinates
            "x_norm": x_norm,
            "y_norm": y_norm,

            # Depth
            "z": z,

            # Confidence
            "visibility": visibility
        }

    return data

```

---

### <a id="📄-core-utils-py"></a>📄 `core/utils.py`

**File Info:**
- **Size**: 0 B
- **Extension**: `.py`
- **Language**: `python`
- **Location**: `core/utils.py`
- **Relative Path**: `core`
- **Created**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **Modified**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **MD5**: `d41d8cd98f00b204e9800998ecf8427e`
- **SHA256**: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`
- **Encoding**: ASCII

**File code content:**

```python

```

---

## 🚫 Binary/Excluded Files

The following files were not included in the text content:

- `models/pose_landmarker_full.task`
- `models/pose_landmarker_heavy.task`
- `models/pose_landmarker_lite.task`

## 🚫 Binary/Excluded Files

The following files were not included in the text content:

- `modes/__pycache__/image_mode.cpython-312.pyc`
- `modes/__pycache__/live_stream.cpython-312.pyc`
- `modes/__pycache__/video_mode.cpython-312.pyc`

### <a id="📄-modes-image-mode-py"></a>📄 `modes/image_mode.py`

**File Info:**
- **Size**: 904 B
- **Extension**: `.py`
- **Language**: `python`
- **Location**: `modes/image_mode.py`
- **Relative Path**: `modes`
- **Created**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **Modified**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **MD5**: `18c0397150cd58fa562cd38d9ad1f610`
- **SHA256**: `b0c42c6f1f8b7a66eb8be094639c452eb5023307a35d3921b386bd77cbe094bf`
- **Encoding**: ASCII

**File code content:**

```python
import cv2

from mediapipe.tasks.python import vision

from core.detector import PoseDetector
from core.landmarks import extract_landmarks
from core.drawing import draw_landmarks_on_image


def run_image_mode(image_path):

    detector = PoseDetector(
        running_mode=vision.RunningMode.IMAGE
    )

    image = cv2.imread(image_path)

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    result = detector.detect_image(rgb_image)

    h, w, _ = image.shape

    landmarks = extract_landmarks(result, w, h)

    print(landmarks)

    annotated_image = draw_landmarks_on_image(
        rgb_image,
        result,
        landmarks
    )

    annotated_image = cv2.cvtColor(
        annotated_image,
        cv2.COLOR_RGB2BGR
    )

    cv2.imshow("Image Mode", annotated_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

```

---

### <a id="📄-modes-live-stream-py"></a>📄 `modes/live_stream.py`

**File Info:**
- **Size**: 2.95 KB
- **Extension**: `.py`
- **Language**: `python`
- **Location**: `modes/live_stream.py`
- **Relative Path**: `modes`
- **Created**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **Modified**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **MD5**: `0c4c8e8b843bf1122bdefc09abec1075`
- **SHA256**: `d421f2f761f62d6f132c86c51c6324a9717d48b947e2e9b89630b46037132bf4`
- **Encoding**: ASCII

**File code content:**

```python
import cv2
import time

from mediapipe.tasks.python import vision

from core.detector import PoseDetector
from core.landmarks import extract_landmarks
from core.drawing import draw_landmarks_on_image

from config.settings import WINDOW_NAME


# ==========================================
# GLOBAL RESULT VARIABLE
# ==========================================

latest_result = None


# ==========================================
# CALLBACK FUNCTION
# ==========================================

def save_result(result, output_image, timestamp_ms):

    global latest_result

    latest_result = result


# ==========================================
# LIVE STREAM MODE
# ==========================================

def run_live_stream():

    global latest_result

    detector = PoseDetector(
        running_mode=vision.RunningMode.LIVE_STREAM,
        result_callback=save_result
    )

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("Cannot open webcam")
        return

    while cap.isOpened():

        success, frame = cap.read()

        if not success:

            print("Ignoring empty frame.")
            continue

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Timestamp required
        timestamp_ms = int(time.time() * 1000)

        # Async detection
        detector.detect_async(
            rgb_frame,
            timestamp_ms
        )

        # Wait for first result
        if latest_result is None:

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            continue

        h, w, _ = frame.shape

        # Extract landmarks
        landmarks = extract_landmarks(
            latest_result,
            w,
            h
        )

        # Print coordinates
        if landmarks:

            print("\n====================")
            print("POSE COORDINATES")
            print("====================")

            for name, point in landmarks.items():

                print(
                    f"{name}: "
                    f"x={point['x']} "
                    f"y={point['y']} "
                    f"z={point['z']:.4f}"
                )

        # Draw skeleton
        annotated_image = draw_landmarks_on_image(
            rgb_frame,
            latest_result,
            landmarks
        )

        # RGB -> BGR
        annotated_image = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_RGB2BGR
        )

        # Show result
        cv2.imshow(
            WINDOW_NAME,
            annotated_image
        )

        # Quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

    cv2.destroyAllWindows()

```

---

### <a id="📄-modes-video-mode-py"></a>📄 `modes/video_mode.py`

**File Info:**
- **Size**: 2.33 KB
- **Extension**: `.py`
- **Language**: `python`
- **Location**: `modes/video_mode.py`
- **Relative Path**: `modes`
- **Created**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **Modified**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **MD5**: `0036ceb6ce84b4852a9171ca0044468f`
- **SHA256**: `5e94769b2df77226ff998fa91c745af8898d1659df78982458c4290eff1a5021`
- **Encoding**: ASCII

**File code content:**

```python
import cv2
import time

from mediapipe.tasks.python import vision

from core.detector import PoseDetector
from core.landmarks import extract_landmarks
from core.drawing import draw_landmarks_on_image


def run_video_mode(video_path):

    # Create detector using VIDEO mode
    detector = PoseDetector(
        running_mode=vision.RunningMode.VIDEO
    )

    # Open video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened correctly
    if not cap.isOpened():
        print("Error opening video.")
        return

    while cap.isOpened():

        # Read frame
        success, frame = cap.read()

        # Stop when video ends
        if not success:
            print("Video finished.")
            break

        # Convert BGR -> RGB
        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # Timestamp required by MediaPipe VIDEO mode
        timestamp_ms = int(time.time() * 1000)

        # Run pose detection
        result = detector.detect_video_frame(
            rgb_frame,
            timestamp_ms
        )

        # Get frame size
        h, w, _ = frame.shape

        # Extract landmarks
        landmarks = extract_landmarks(
            result,
            w,
            h
        )

        # Print coordinates
        if landmarks:

            print("\n====================")
            print("POSE COORDINATES")
            print("====================")

            for name, point in landmarks.items():

                print(
                    f"{name}: "
                    f"x={point['x']} "
                    f"y={point['y']} "
                    f"z={point['z']:.4f}"
                )

        # Draw skeleton + angles
        annotated_image = draw_landmarks_on_image(
            rgb_frame,
            result,
            landmarks
        )

        # Convert RGB -> BGR for OpenCV display
        annotated_image = cv2.cvtColor(
            annotated_image,
            cv2.COLOR_RGB2BGR
        )

        # Show video
        cv2.imshow(
            "Basketball Pose Detection",
            annotated_image
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

```

---

### <a id="📄-main-py"></a>📄 `main.py`

**File Info:**
- **Size**: 638 B
- **Extension**: `.py`
- **Language**: `python`
- **Location**: `main.py`
- **Relative Path**: `root`
- **Created**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **Modified**: 2026-06-22 05:08:41 (Asia/Damascus / GMT+03:00)
- **MD5**: `c491caae79d166bf749abadc4d012439`
- **SHA256**: `741e4a8869ecf09560b19a3237d02f51c29eb24851f8ddf3d9b295b0e7f3af74`
- **Encoding**: ASCII

**File code content:**

```python
from modes.live_stream import run_live_stream
from modes.image_mode import run_image_mode
from modes.video_mode import run_video_mode


# ==================================
# SELECT MODE
# ==================================

MODE = "image"

# MODE = "image"
# MODE = "video"


# ==================================
# RUN
# ==================================

if MODE == "live":

    run_live_stream()

elif MODE == "image":

    run_image_mode(
        "assets/test.jpg"
    )

elif MODE == "video":

    run_video_mode(
        "assets/Untitled.mov"
    )

else:

    print("Invalid mode selected.")

```

---

### <a id="📄-readme-md"></a>📄 `README.md`

**File Info:**
- **Size**: 2.75 KB
- **Extension**: `.md`
- **Language**: `text`
- **Location**: `README.md`
- **Relative Path**: `root`
- **Created**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **Modified**: 2026-05-27 15:32:14 (Asia/Damascus / GMT+03:00)
- **MD5**: `514153462b0a4ca8ea4b9e74a86bd6ff`
- **SHA256**: `d6cd38284e5113163671e5226419c8a06c3e0d921d7fa0fa313be6ad5bb2fe75`
- **Encoding**: UTF-8

**File code content:**

````markdown
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

~~~~bash
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
~~~~

---

## ⚙️ Installation

### 1️⃣ Clone the repository

~~~~bash
git clone https://github.com/yourusername/swichy.git
cd swichy
~~~~

### 2️⃣ Create virtual environment

~~~~bash
python -m venv venv
~~~~

### 3️⃣ Activate virtual environment

#### Windows

~~~~bash
venv\Scripts\activate
~~~~

#### Linux / macOS

~~~~bash
source venv/bin/activate
~~~~

### 4️⃣ Install dependencies

~~~~bash
pip install mediapipe opencv-python numpy
~~~~

Or using `requirements.txt`:

~~~~bash
pip install -r requirements.txt
~~~~

---

## 📦 Required Libraries

- `mediapipe` → AI pose estimation
- `opencv-python` → Computer vision & webcam processing
- `numpy` → Mathematical operations

---

## ▶️ Run Swichy

~~~~bash
python main.py
~~~~

---

## 🛠 Example requirements.txt

~~~~txt
mediapipe
opencv-python
numpy
~~~~

````

---

## 🚫 Binary/Excluded Files

The following files were not included in the text content:

- `main.ipynb`


"""Load application settings and YAML configuration paths."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "models" / "pose_landmarker_lite.task")

WINDOW_NAME = "Swichy — AI Basketball Coach"

# MediaPipe confidence thresholds
MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
MIN_PRESENCE_CONFIDENCE = 0.5

# Visibility gating (visibility + presence)
VISIBILITY_THRESHOLD = 0.6
PRESENCE_THRESHOLD = 0.5
VISIBILITY_HOLD_FRAMES = 5
VISIBILITY_REQUIRE_PRESENCE = True

# One Euro filter defaults
FILTER_MIN_CUTOFF = 1.0
FILTER_BETA = 0.007
FILTER_D_CUTOFF = 1.0

# Shooting analysis
SHOOTING_HAND = "auto"  # auto | left | right
DEFAULT_FPS = 30.0

# Frame buffer
FRAME_BUFFER_SIZE = 300

# Report output
REPORT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "reports")

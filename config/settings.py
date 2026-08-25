"""Load application settings and YAML configuration paths."""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSE_MODEL = "lite"          # lite | full | heavy
MODEL_PATH = str(PROJECT_ROOT / "models" / f"pose_landmarker_{POSE_MODEL}.task")

# CHANGING `POSE_MODEL` CHANGES EVERY SCORE, and not by a rounding amount.
# Measured on the same footage, same rules, only the model swapped:
#
#     clip                   lite   full   heavy
#     Klay3ptshooting1         89     72      77
#     Klay3ptshooting3         89     78      82
#     Klay3ptshooting4         89     92      82
#     mean over 12 shots     90.1   84.4    79.8
#
# Up to 17 points on identical frames. A rule band is therefore only valid for
# the model it was derived against, which is why `biomechanics.yaml` carries a
# `calibration:` block per model and the engine selects the one matching this
# setting. Switching the model without a calibration block for it falls back to
# the rule's top-level numbers, and those describe `lite`.
#
# `lite` is the least accurate of the three. Rigid-bone variation on video8,
# lower is better:
#
#     model    upper arm  forearm  thigh  shank
#     lite         18.28    21.64  18.73  11.02
#     full         10.46    17.51   9.28   6.64
#     heavy        17.41    19.04   8.39   6.59
#
# `full` reconstructs the arm best -- 43% better than lite at the upper arm --
# and the arm is what a shooting rule reads. `heavy` is NOT the upgrade: it is
# worse than `full` on both arm segments, which docs/LIMITS.md also records.

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

# One Euro filter defaults. Fallbacks for config/filter_config.yaml -- keep the
# two in step, or a missing key silently restores a value nobody chose.
# beta is metre-scale, because world landmarks are in metres; the pixel-scale
# 0.007 that used to sit here made the filter non-adaptive. See the yaml.
FILTER_MIN_CUTOFF = 1.0
FILTER_BETA = 10.0
FILTER_D_CUTOFF = 1.0

# Shooting analysis
SHOOTING_HAND = "auto"  # auto | left | right
DEFAULT_FPS = 30.0

# Frame buffer
FRAME_BUFFER_SIZE = 300

# Report output
REPORT_OUTPUT_DIR = str(PROJECT_ROOT / "outputs" / "reports")

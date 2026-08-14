"""Live camera view of the joint angles. A measuring instrument, nothing else.

    python scripts/live_angles.py            # default camera
    python scripts/live_angles.py --camera 1 # a different one
    python scripts/live_angles.py --mirror   # flip, so it feels like a mirror

    Q or ESC quit    SPACE freeze/unfreeze    H hide the skeleton

WHAT THIS IS FOR
Checking that MediaPipe returns angles that match what your body is doing. Hold
your arm straight and the elbow should read near 180; bend it square and it
should read near 90. If those two do not hold, nothing downstream can be
trusted, and this is the fastest way to find that out.

WHY IT IS SEPARATE FROM EVERYTHING ELSE
It shares the angle maths (`angles/`, `geometry/`) and nothing else. No shot
detection, no segmentation, no scoring, no offline pass -- so what you see here
cannot be an artefact of any of them. If an angle looks wrong here, it is the
pose or the angle code; if it looks right here and wrong in a report, it is
something after this point. That separation is the whole value.

REAL TIME IS FINE HERE, AND ONLY HERE
The main pipeline is deliberately offline, because a shot's phases cannot be
known until the shot is over. An angle has no such problem: it is a property of
a single frame, complete the moment that frame arrives.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
from mediapipe.tasks.python import vision  # noqa: E402

from angles.calculator import AngleCalculator  # noqa: E402
from config.settings import (  # noqa: E402
    PRESENCE_THRESHOLD,
    VISIBILITY_HOLD_FRAMES,
    VISIBILITY_REQUIRE_PRESENCE,
    VISIBILITY_THRESHOLD,
)
from pose.detector import PoseDetector  # noqa: E402
from pose.landmarks import extract_all_landmarks  # noqa: E402
from pose.visibility import VisibilityGate  # noqa: E402

# Drawn pairs. Face points are left out: nothing here is measured from them.
CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32), (15, 19), (16, 20),
)

# What to show, and the range each should sit in when you are standing
# normally. The expected range is the point: a number alone tells you nothing
# about whether it is right.
ANGLES = (
    ("elbow", "Elbow", "straight ~180, square ~90"),
    ("shoulder", "Shoulder", "arm down ~10, out ~90"),
    ("knee", "Knee", "standing ~175, squat ~90"),
    ("hip", "Hip", "standing ~175, hinged ~120"),
    ("index_align", "Finger line", "aligned ~180"),
)


def _text(img, s, x, y, colour=(255, 255, 255), scale=0.55, weight=1):
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                weight + 2, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                weight, cv2.LINE_AA)


def _panel(img, x1, y1, x2, y2, alpha=0.6):
    patch = img[y1:y2, x1:x2]
    if patch.size:
        img[y1:y2, x1:x2] = cv2.addWeighted(patch, 1 - alpha, patch * 0, alpha, 0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--side", default="right", choices=["right", "left"])
    parser.add_argument("--mirror", action="store_true",
                        help="flip horizontally so the view behaves like a mirror")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Cannot open camera {args.camera}. Try --camera 1.")
        return 2

    detector = PoseDetector(vision.RunningMode.VIDEO)
    # The same gate the pipeline uses, with the same settings, so an angle
    # shown here is judged reliable by exactly the rule that governs it in a
    # report. A tool that measured under laxer rules would be worse than no
    # tool: it would clear a joint the pipeline goes on to reject.
    gate = VisibilityGate(
        visibility_threshold=VISIBILITY_THRESHOLD,
        presence_threshold=PRESENCE_THRESHOLD,
        hold_frames=VISIBILITY_HOLD_FRAMES,
        require_presence=VISIBILITY_REQUIRE_PRESENCE,
    )
    calculator = AngleCalculator(gate)

    print("\n  Live angle check. Q quit | SPACE freeze | H hide skeleton\n")
    print("  Sanity test: straighten your arm (elbow -> ~180), then bend it")
    print("  square (elbow -> ~90). If those hold, the angle chain is sound.\n")

    frame_index, frozen, show_skeleton = 0, False, True
    last = None

    while True:
        if not frozen:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            last = frame.copy()

            # A monotonic timestamp from the frame counter, not the wall clock:
            # MediaPipe's VIDEO mode requires monotonic input, and a clock can
            # stall or repeat under load.
            timestamp_ms = int(frame_index * 33)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_video_frame(rgb, timestamp_ms)
            frame_index += 1
        else:
            frame = last.copy()
            result = detector.detect_video_frame(
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), int(frame_index * 33)
            )

        h, w = frame.shape[:2]
        raw = extract_all_landmarks(result, w, h)

        if raw is None:
            _panel(frame, 0, 0, 360, 60)
            _text(frame, "No person detected", 14, 38, (60, 90, 255), 0.7, 2)
        else:
            # Same order as pipeline.process_frame: gate the landmarks, then
            # measure. The gate marks each landmark reliable or not, and the
            # calculator refuses to build an angle from an unreliable one --
            # skip it and every angle comes back invalid.
            #
            # The One Euro filter the pipeline also runs is deliberately NOT
            # applied here. It smooths across frames, and smoothing is exactly
            # what you do not want when the question is "does this instant read
            # correctly?" -- it would hide the jitter this tool exists to show.
            world = gate.apply(raw["world"])
            angles = calculator.compute_all(world, args.side)

            if show_skeleton and result.pose_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h))
                       for lm in result.pose_landmarks[0]]
                for a, b in CONNECTIONS:
                    if a < len(pts) and b < len(pts):
                        cv2.line(frame, pts[a], pts[b], (0, 220, 120), 2, cv2.LINE_AA)
                for p in pts:
                    cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)

            rows = []
            for key, label, hint in ANGLES:
                res = angles.get(f"{args.side}_{key}") or angles.get(key)
                if res is None or not res.is_valid or res.degrees is None:
                    rows.append((label, "--", hint, (120, 120, 120)))
                else:
                    colour = (255, 255, 255) if res.is_stable else (0, 200, 255)
                    rows.append((label, f"{res.degrees:6.1f}", hint, colour))

            _panel(frame, 0, 0, 470, 56 + 30 * len(rows))
            _text(frame, f"{args.side.upper()} SIDE   (world-space 3D angles)",
                  14, 30, (180, 180, 180), 0.55)
            y = 60
            for label, value, hint, colour in rows:
                _text(frame, label, 14, y, (200, 200, 200), 0.55)
                _text(frame, value, 130, y, colour, 0.62, 2)
                _text(frame, hint, 210, y, (130, 130, 130), 0.42)
                y += 30

            _text(frame, "orange = unstable this frame", 14, y + 4,
                  (0, 200, 255), 0.42)

        if frozen:
            _text(frame, "FROZEN", w // 2 - 50, 36, (0, 210, 255), 0.75, 2)

        cv2.imshow("Swichy - live angle check", frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            frozen = not frozen
        elif key == ord("h"):
            show_skeleton = not show_skeleton

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

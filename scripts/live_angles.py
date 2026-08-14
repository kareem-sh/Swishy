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

NOTHING TEMPORAL IS APPLIED, AND THAT TOOK A DELIBERATE CHOICE
Two mechanisms in the pipeline reach across frames, and both are off here:

  * the One Euro filter, which smooths landmark positions over time;
  * the visibility gate's HOLD, which -- when a landmark's confidence dips --
    substitutes its position from an earlier frame for up to `hold_frames`.

The hold is the dangerous one for a measuring instrument. It does not smooth a
value, it FREEZES one: the number on screen belongs to an earlier moment while
looking exactly like a current reading. Measured over 1362 frames of
salah_video.mp4, held readings were 0.4% of the total but wrong by up to 128
degrees against the same frame's raw MediaPipe angle. On the other 99.6% this
tool's output matched raw MediaPipe to 0.00000000 degrees.

So the gate here runs with hold_frames=0. Thresholds still apply, because an
angle the pipeline would reject must not look trustworthy here -- but a
landmark that fails them reads "--" rather than a stale number. That is the
project's standing rule: never silently turn "not observed" into a value.
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
from angles.joint_chains import JOINT_CHAINS  # noqa: E402
from config.settings import (  # noqa: E402
    PRESENCE_THRESHOLD,
    VISIBILITY_HOLD_FRAMES,
    VISIBILITY_REQUIRE_PRESENCE,
    VISIBILITY_THRESHOLD,
)
from pose.detector import PoseDetector  # noqa: E402
from pose.landmarks import POSE_LANDMARKS, extract_all_landmarks  # noqa: E402
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
#
# The ranges are RANGES, not targets. A fully extended human elbow does not
# reach 180: the carrying angle holds the forearm a few degrees off the line of
# the upper arm, so 170-180 is a straight arm and reading 175 is not an error of
# 5. Quoting a single number here would invite exactly that misreading.
ANGLES = (
    ("elbow", "Elbow", "straight 170-180, square ~90"),
    ("shoulder", "Shoulder", "down ~10, out ~90  (+-26 deg, see SOURCES D4)"),
    ("knee", "Knee", "standing 170-180, squat ~90"),
    ("hip", "Hip", "standing 170-180, hinged ~120"),
    ("index_align", "Finger line", "aligned 160-180"),
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


def _flat_angle(pts, a, b, c):
    """The angle as it appears on screen, in pixels. No depth involved.

    Pixel coordinates, not the normalised ones: x and y are normalised by width
    and height separately, so on a 16:9 frame an angle computed from them is
    distorted by the aspect ratio before anyone looks at it.
    """
    import math
    (ax, ay), (bx, by), (cx, cy) = pts[a], pts[b], pts[c]
    v1, v2 = (ax - bx, ay - by), (cx - bx, cy - by)
    n1 = math.hypot(*v1)
    n2 = math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _draw_arc(img, pts, a, b, c, colour, radius=38):
    """Draw the angle where it lives, so the number can be eyeballed."""
    import math
    (ax, ay), (bx, by), (cx, cy) = pts[a], pts[b], pts[c]
    t1 = math.degrees(math.atan2(ay - by, ax - bx))
    t2 = math.degrees(math.atan2(cy - by, cx - bx))
    sweep = (t2 - t1 + 180) % 360 - 180          # shorter way round
    cv2.ellipse(img, (bx, by), (radius, radius), 0, t1, t1 + sweep, colour, 2,
                cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--video", default=None,
                        help="read a video file instead of the camera, to check the "
                             "angles on the footage that is actually analysed")
    parser.add_argument("--side", default="right", choices=["right", "left"])
    parser.add_argument("--mirror", action="store_true",
                        help="flip horizontally so the view behaves like a mirror")
    parser.add_argument("--pipeline-hold", action="store_true",
                        help="re-enable the visibility gate's temporal hold, to see "
                             "what the pipeline sees rather than what the camera sees")
    args = parser.parse_args()

    if args.video:
        source = Path(args.video)
        if not source.exists():
            print(f"No such video: {source}")
            return 2
        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            print(f"Cannot open {source}.")
            return 2
    else:
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"Cannot open camera {args.camera}. Try --camera 1.")
            return 2

    detector = PoseDetector(vision.RunningMode.VIDEO)
    # The pipeline's gate, with the pipeline's THRESHOLDS -- so a joint the
    # pipeline would reject cannot look trustworthy here -- but with the
    # temporal hold switched off (see the module docstring). Holding replaces a
    # dipped landmark with its position from an earlier frame, which is exactly
    # the fabrication this tool exists to expose, not to commit.
    gate = VisibilityGate(
        visibility_threshold=VISIBILITY_THRESHOLD,
        presence_threshold=PRESENCE_THRESHOLD,
        hold_frames=0 if not args.pipeline_hold else VISIBILITY_HOLD_FRAMES,
        require_presence=VISIBILITY_REQUIRE_PRESENCE,
    )
    calculator = AngleCalculator(gate)

    if args.video:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step_ms = 1000.0 / fps
        print(f"\n  Angle check on {Path(args.video).name}  "
              f"({total} frames @ {fps:.1f} fps)")
        print("  SPACE pause | A/D step frame by frame when paused | "
              "H hide skeleton | Q quit\n")
        print("  The big number is the 3D world angle -- the one that is scored.")
        print("  'flat' beside it is the same angle measured in the image, with no")
        print("  depth at all. Where the two disagree, the gap IS the depth guess.\n")
    else:
        fps, total, step_ms = 30.0, 0, 33.0
        print("\n  Live angle check. Q quit | SPACE freeze | H hide skeleton\n")
        print("  Sanity test: straighten your arm (elbow -> 170-180), then bend it")
        print("  square (elbow -> ~90). If those hold, the angle chain is sound.\n")

    frame_index, frozen, show_skeleton, pause_next = 0, False, True, False
    detect_ms = 0          # never decreases, even when the video is rewound
    last = last_result = None

    while True:
        if not frozen:
            ok, frame = cap.read()
            if not ok:
                break
            if args.mirror:
                frame = cv2.flip(frame, 1)
            last = frame.copy()

            # The detector's clock only ever moves forward. MediaPipe's VIDEO
            # mode requires strictly increasing timestamps, and stepping
            # backwards through a file would otherwise hand it a smaller one.
            # This clock is for the detector alone; frame_index is what the
            # display and the seeking use.
            detect_ms += int(step_ms) or 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = last_result = detector.detect_video_frame(rgb, detect_ms)
            frame_index += 1
            if pause_next:
                frozen, pause_next = True, False
        else:
            # Reuse the cached result rather than re-detecting. MediaPipe's
            # VIDEO mode requires strictly increasing timestamps, so re-running
            # a held frame either errors or silently re-times the stream.
            frame = last.copy()
            result = last_result

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

            pts = None
            if result.pose_landmarks:
                pts = [(int(lm.x * w), int(lm.y * h))
                       for lm in result.pose_landmarks[0]]

            if show_skeleton and pts:
                for a, b in CONNECTIONS:
                    if a < len(pts) and b < len(pts):
                        cv2.line(frame, pts[a], pts[b], (0, 220, 120), 2, cv2.LINE_AA)
                for p in pts:
                    cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)

            rows = []
            for key, label, hint in ANGLES:
                name = f"{args.side}_{key}"
                res = angles.get(name) or angles.get(key)
                if res is None or not res.is_valid or res.degrees is None:
                    rows.append((label, "--", "", hint, (120, 120, 120)))
                    continue
                colour = (255, 255, 255) if res.is_stable else (0, 200, 255)

                # The same angle with the depth axis thrown away. It is not
                # more correct -- a limb pointing at the camera genuinely
                # foreshortens -- but it is computed from pixels you can see,
                # so a large gap tells you the 3D value rests on a depth guess
                # rather than on anything visible in the frame.
                flat = ""
                chain = JOINT_CHAINS.get(name)
                if chain and pts:
                    ids = [POSE_LANDMARKS.get(n) for n in chain.landmark_names]
                    if all(i is not None and i < len(pts) for i in ids):
                        f = _flat_angle(pts, *ids)
                        if f is not None:
                            flat = f"flat {f:5.1f}"
                            if show_skeleton:
                                _draw_arc(frame, pts, *ids, colour)
                                _text(frame, f"{res.degrees:.0f}",
                                      pts[ids[1]][0] + 42, pts[ids[1]][1] - 6,
                                      colour, 0.5, 2)
                rows.append((label, f"{res.degrees:6.1f}", flat, hint, colour))

            _panel(frame, 0, 0, 560, 56 + 30 * len(rows))
            _text(frame, f"{args.side.upper()} SIDE   (world-space 3D angles)",
                  14, 30, (180, 180, 180), 0.55)
            y = 60
            for label, value, flat, hint, colour in rows:
                _text(frame, label, 14, y, (200, 200, 200), 0.55)
                _text(frame, value, 130, y, colour, 0.62, 2)
                _text(frame, flat, 205, y, (150, 150, 150), 0.45)
                _text(frame, hint, 300, y, (130, 130, 130), 0.42)
                y += 30

            # With the hold off, an untrusted landmark yields "--" and orange
            # cannot occur -- so say what "--" means instead of describing a
            # colour that will never appear.
            if args.pipeline_hold:
                _text(frame, "orange = HELD from an earlier frame, not measured now",
                      14, y + 4, (0, 200, 255), 0.42)
            else:
                _text(frame, '"--" = not measurable this frame (nothing is held)',
                      14, y + 4, (140, 140, 140), 0.42)

        if args.video:
            _text(frame, f"{frame_index * step_ms / 1000:6.2f}s  "
                         f"frame {frame_index}/{total}", w - 260, 30,
                  (200, 200, 200), 0.55)
        if frozen:
            _text(frame, "PAUSED", w // 2 - 50, 36, (0, 210, 255), 0.75, 2)

        title = "Swichy - angle check" + (f" - {Path(args.video).name}"
                                          if args.video else " (live)")
        cv2.imshow(title, frame)
        wait = 0 if (frozen and args.video) else (
            max(1, int(step_ms)) if args.video else 1)
        key = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            frozen = not frozen
        elif key == ord("h"):
            show_skeleton = not show_skeleton
        elif args.video and key in (ord("d"), 83):
            frozen, pause_next = False, True          # one frame, then hold
        elif args.video and key in (ord("a"), 81):
            frame_index = max(0, frame_index - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            frozen, pause_next = False, True

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

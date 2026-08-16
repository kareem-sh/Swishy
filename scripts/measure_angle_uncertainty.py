"""How wrong can a joint angle be? Bound it without any ground truth.

    python scripts/measure_angle_uncertainty.py
    python scripts/measure_angle_uncertainty.py --video assets/videos/video8.mov

THE PROBLEM WITH ASKING "IS THIS ANGLE CORRECT"
There is no protractor in the footage. Nothing in the video states what the
elbow angle truly was, so accuracy cannot be measured directly, and any claim
that the angles are "correct" would be an opinion wearing a number's clothes.

WHAT CAN BE MEASURED INSTEAD
Two things, neither of which needs truth:

  MODEL DISAGREEMENT. Run the same frames through MediaPipe's lite, full and
  heavy models. All three are Google's, trained for this task, equally
  plausible a priori. Wherever two of them differ, at least one is wrong by
  that much -- so their disagreement is a LOWER BOUND on the error of whichever
  one is shipped. It cannot prove an angle right. It can prove how wrong an
  angle might be, which is the more useful half.

  RIGID-BONE CONSISTENCY. A humerus does not change length. Every metre of
  variation in a reconstructed bone is reconstruction error, expressed in
  metres rather than opinions, and an angle is only as good as the points it
  is built from.

WHY IT MATTERS FOR SCORING
A rule band narrower than this uncertainty is scoring noise. The numbers this
prints are what `config/biomechanics.yaml` band widths must be judged against,
and what docs/LIMITS.md reports. Re-run it if the model, the footage or the
filter changes.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.tasks.python import BaseOptions, vision  # noqa: E402

MODELS = {
    "lite": "models/pose_landmarker_lite.task",
    "full": "models/pose_landmarker_full.task",
    "heavy": "models/pose_landmarker_heavy.task",
}
# Raw MediaPipe indices, deliberately not the project's named landmarks: this
# script must stay independent of the code it is auditing.
IDX = {"right_shoulder": 12, "right_elbow": 14, "right_wrist": 16,
       "right_hip": 24, "right_knee": 26, "right_ankle": 28,
       "right_index": 20}
CHAINS = {
    "elbow": ("right_shoulder", "right_elbow", "right_wrist"),
    "knee": ("right_hip", "right_knee", "right_ankle"),
    "hip": ("right_shoulder", "right_hip", "right_knee"),
    "shoulder": ("right_hip", "right_shoulder", "right_elbow"),
    # Added 16 Aug 2026. `index_align` is the WRIST angle -- elbow, wrist,
    # index -- and two scored rules now depend on it, but it had never been
    # put through this audit: the 39.2 deg figure quoted in biomechanics.yaml
    # came from elsewhere and was never reproduced here. A band was being
    # derived from a number this tool had not produced.
    "index_align": ("right_elbow", "right_wrist", "right_index"),
}
BONES = {"upper arm": ("right_shoulder", "right_elbow"),
         "forearm": ("right_elbow", "right_wrist"),
         "thigh": ("right_hip", "right_knee"),
         "shank": ("right_knee", "right_ankle")}


def _angle(pose, a, b, c):
    pa, pb, pc = pose[IDX[a]], pose[IDX[b]], pose[IDX[c]]
    v1 = np.array([pa.x - pb.x, pa.y - pb.y, pa.z - pb.z])
    v2 = np.array([pc.x - pb.x, pc.y - pb.y, pc.z - pb.z])
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def _run(model_path: Path, video: Path, every: int) -> dict:
    landmarker = vision.PoseLandmarker.create_from_options(
        vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO))
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out, index = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        index += 1
        if index % every:
            continue
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(image, int(index * 1000.0 / fps))
        if not result.pose_world_landmarks:
            continue
        pose = result.pose_world_landmarks[0]
        out[index] = {
            "angles": {k: _angle(pose, *v) for k, v in CHAINS.items()},
            "bones": {k: math.dist(
                (pose[IDX[a]].x, pose[IDX[a]].y, pose[IDX[a]].z),
                (pose[IDX[b]].x, pose[IDX[b]].y, pose[IDX[b]].z))
                for k, (a, b) in BONES.items()},
            "vis": min(pose[IDX[n]].visibility for n in IDX),
        }
    cap.release()
    landmarker.close()
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", default="assets/videos/salah_video.mp4")
    parser.add_argument("--every", type=int, default=3,
                        help="sample every Nth frame; heavy is slow and every "
                             "third frame is ample for a distribution")
    args = parser.parse_args()

    video = PROJECT_ROOT / args.video
    if not video.exists():
        video = Path(args.video)
    if not video.exists():
        print(f"No such video: {args.video}")
        return 2

    data = {}
    for name, rel in MODELS.items():
        path = PROJECT_ROOT / rel
        if not path.exists():
            print(f"  missing model, skipping: {rel}")
            continue
        print(f"  running {name} ...", flush=True)
        data[name] = _run(path, video, args.every)

    if len(data) < 2:
        print("Need at least two models to bound anything.")
        return 1

    print(f"\n{video.name}   every {args.every}th frame\n")
    print("=" * 72)
    print("RIGID BONES -- coefficient of variation %, lower is better")
    print("=" * 72)
    print(f"  {'model':<8}" + "".join(f"{b:>12}" for b in BONES))
    for name, frames in data.items():
        row = []
        for bone in BONES:
            v = np.array([f["bones"][bone] for f in frames.values()])
            row.append(100 * v.std() / v.mean() if v.size else float("nan"))
        print(f"  {name:<8}" + "".join(f"{c:12.2f}" for c in row))

    print()
    print("=" * 72)
    print("MODEL DISAGREEMENT on the same frame, degrees")
    print("=" * 72)
    print("A lower bound on the error of whichever model ships.")
    for a, b in (("full", "heavy"), ("lite", "full")):
        if a not in data or b not in data:
            continue
        shared = sorted(set(data[a]) & set(data[b]))
        for label, subset in (
            (f"{a} vs {b}", shared),
            ("...on high-confidence frames only (all visibility > 0.9)",
             [i for i in shared
              if data[a][i]["vis"] > 0.9 and data[b][i]["vis"] > 0.9]),
        ):
            if not subset:
                continue
            print(f"\n  {label}   ({len(subset)} frames)")
            print(f"    {'angle':<12}{'mean':>9}{'median':>9}{'p90':>9}"
                  f"{'max':>9}{'>10deg':>9}")
            for key in CHAINS:
                d = [abs(data[a][i]["angles"][key] - data[b][i]["angles"][key])
                     for i in subset
                     if data[a][i]["angles"][key] is not None
                     and data[b][i]["angles"][key] is not None]
                if not d:
                    continue
                d = np.array(d)
                print(f"    {key:<12}{d.mean():9.2f}{np.median(d):9.2f}"
                      f"{np.percentile(d, 90):9.2f}{d.max():9.2f}"
                      f"{100 * np.mean(d > 10):8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

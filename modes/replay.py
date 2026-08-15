"""Play a video back with the finished analysis drawn on it.

TWO PASSES, AND WHY THAT IS THE POINT

Pass one reads the file and analyses it. Pass two reads the same file again and
draws. Nothing is shown until the analysis is complete.

That is not a limitation to apologise for -- it is the reason the overlay can
be correct. A frame's coaching phase is decided by where the knee reached its
lowest angle and where the hand reached its highest point, and neither is known
while that frame is on screen. A live overlay can only ever show a guess that
later turns out wrong; this one shows the answer.

Pose detection is not repeated. The landmarks from pass one are kept and
redrawn, so the second pass costs only video decoding.

EVERYTHING IS ADDRESSED BY TIMESTAMP

Not by list position. Frames where MediaPipe found nobody are absent from the
analysis history, so its indices are not video frame numbers -- and drawing
frame N's skeleton over frame N+k's pixels is exactly how the overlay came to
run ahead of the player. The timestamp is the only identifier both passes agree
on.
"""

from __future__ import annotations

from bisect import bisect_left
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

from visualization.renderer import draw_ball_overlays

# MediaPipe's 33-point skeleton, as the pairs worth drawing. Face points are
# omitted: they add clutter and nothing here is measured from them.
CONNECTIONS: Sequence[Tuple[int, int]] = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32), (15, 19), (16, 20),
)

PHASE_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "loading": (80, 180, 255),
    "ball_lift": (0, 210, 255),
    "jump": (0, 230, 120),
    "release": (60, 90, 255),
    "follow_through": (255, 170, 60),
    "landing": (200, 120, 255),
}
IDLE = (90, 90, 90)

# Which angles to show, and what to call them on screen.
ANGLE_LABELS = (
    ("elbow", "Elbow"),
    ("knee", "Knee"),
    ("hip", "Hip"),
    ("shoulder", "Shoulder"),
    ("index_align", "Finger"),
    ("trunk", "Trunk lean"),
)


def _text(frame, s, x, y, colour=(255, 255, 255), scale=0.55, weight=1):
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                weight + 2, cv2.LINE_AA)
    cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                weight, cv2.LINE_AA)


def _shade(frame, x1, y1, x2, y2, alpha=0.55):
    patch = frame[y1:y2, x1:x2]
    if patch.size:
        frame[y1:y2, x1:x2] = cv2.addWeighted(
            patch, 1 - alpha, patch * 0, alpha, 0
        )


def _draw_angles(frame, angles: Dict[str, float], side: str) -> None:
    """Right-hand column of joint angles, in degrees."""
    if not angles:
        return
    rows = [(label, angles.get(f"{side}_{key}", angles.get(key)))
            for key, label in ANGLE_LABELS]
    rows = [(l, v) for l, v in rows if v is not None]
    if not rows:
        return

    w = frame.shape[1]
    x1, y1 = w - 210, 50
    x2, y2 = w - 10, 62 + 24 * len(rows)
    _shade(frame, x1, y1, x2, y2)
    _text(frame, "ANGLES", x1 + 12, y1 + 22, (180, 180, 180), 0.5)
    y = y1 + 46
    for label, value in rows:
        _text(frame, label, x1 + 12, y, (190, 190, 190), 0.5)
        _text(frame, f"{value:5.1f}", x1 + 135, y, (255, 255, 255), 0.55)
        y += 24


def _draw_timeline(frame, ts_ms, duration_ms, overlay) -> None:
    h, w = frame.shape[:2]
    top, bar = h - 26, 14
    cv2.rectangle(frame, (0, top), (w, top + bar), (25, 25, 25), -1)
    for t, entry in overlay.items():
        if duration_ms <= 0:
            continue
        x = int(w * t / duration_ms)
        cv2.line(frame, (x, top), (x, top + bar),
                 PHASE_COLOURS.get(entry["phase"], IDLE), 1)
    if duration_ms > 0:
        x = int(w * ts_ms / duration_ms)
        cv2.line(frame, (x, top - 4), (x, top + bar + 4), (255, 255, 255), 2)


def replay(
    video_path,
    landmarks: Sequence[Tuple[int, Optional[List[Tuple[float, float]]]]],
    overlay: Dict[int, dict],
    shot_count: int = 0,
    shooting_side: str = "right",
    window_name: str = "Swichy - analysed",
    ball_overlay: Optional[Dict[int, object]] = None,
) -> None:
    """Show the completed pose and basketball analysis on the source video.

    Controls: SPACE pause/resume, A/D or LEFT/RIGHT step while paused, Q quit.
    """
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Cannot open {path} for playback.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(landmarks)
    duration_ms = int(total * 1000.0 / fps) if fps else 0
    delay = max(1, int(1000.0 / fps))

    # Timestamp -> nearest analysed entry, resolved once rather than per frame.
    keys = sorted(overlay)
    ball_overlay = ball_overlay or {}
    ball_keys = sorted(ball_overlay)

    def entry_for(ts: int) -> Optional[dict]:
        if not keys:
            return None
        best = min(keys, key=lambda k: abs(k - ts))
        return overlay[best] if abs(best - ts) <= (1000.0 / fps) else None

    def ball_entry_for(ts: int):
        if not ball_keys:
            return None
        # Frame timestamps are normally exact. The nearest-frame fallback
        # also handles videos whose reported FPS introduces 1 ms rounding.
        exact = ball_overlay.get(ts)
        if exact is not None:
            return exact
        position = bisect_left(ball_keys, ts)
        neighbours = ball_keys[max(0, position - 1):position + 1]
        if not neighbours:
            return None
        best = min(neighbours, key=lambda k: abs(k - ts))
        return (
            ball_overlay[best]
            if abs(best - ts) <= (1000.0 / fps)
            else None
        )

    print(f"\n  Playing back {path.name} with the analysis drawn on it.")
    print("  SPACE pause/resume | A/D step when paused | Q quit\n")

    index, paused = 0, False
    while 0 <= index < total:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            break

        # The timestamp of THIS frame, derived the same way pass one derived
        # it, which is what keeps the two aligned.
        ts = int(index * 1000.0 / fps)
        marks = None
        if index < len(landmarks):
            stamped, candidate = landmarks[index]
            if abs(stamped - ts) <= (1000.0 / fps):
                marks = candidate

        info = entry_for(ts)
        phase = info["phase"] if info else None
        colour = PHASE_COLOURS.get(phase, IDLE)

        ball_info = ball_entry_for(ts)
        if ball_info is not None:
            # The shared renderer draws on RGB images because video mode feeds
            # it RGB. Replay decodes BGR, so convert around only this layer and
            # keep the replay's existing BGR pose colours unchanged.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            draw_ball_overlays(rgb_frame, ball_info)
            frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        if marks:
            h, w = frame.shape[:2]
            pts = [(int(x * w), int(y * h)) for x, y in marks]
            for a, b in CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], colour, 2, cv2.LINE_AA)
            for i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
                if i < len(pts):
                    cv2.circle(frame, pts[i], 4, (255, 255, 255), -1, cv2.LINE_AA)

        _shade(frame, 0, 0, 330, 104)
        _text(frame, f"{ts / 1000:6.2f}s   frame {index}/{total}", 12, 26,
              (200, 200, 200), 0.55)
        if info:
            _text(frame, f"SHOT {info['shot']}  {phase.replace('_', ' ').upper()}",
                  12, 54, colour, 0.62, 2)
            if info["score"] is not None:
                _text(frame, f"score {info['score']}/100", 12, 82)
            _draw_angles(frame, info.get("angles") or {}, shooting_side)
        else:
            _text(frame, "no shot here", 12, 54, IDLE, 0.6)

        _draw_timeline(frame, ts, duration_ms, overlay)
        if shot_count:
            _text(frame, f"{shot_count} shot(s)", frame.shape[1] - 150, 30)
        if paused:
            _text(frame, "PAUSED", frame.shape[1] // 2 - 45, 34,
                  (0, 210, 255), 0.7, 2)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif key in (ord("a"), 81):
            index = max(0, index - 1)
            paused = True
            continue
        elif key in (ord("d"), 83):
            index = min(total - 1, index + 1)
            paused = True
            continue
        if not paused:
            index += 1

    cap.release()
    cv2.destroyAllWindows()

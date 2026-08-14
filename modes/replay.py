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
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2

# MediaPipe's 33-point skeleton, as the pairs worth drawing. Face points are
# omitted: they add clutter and nothing here is measured from them.
CONNECTIONS: Sequence[Tuple[int, int]] = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32), (15, 19), (16, 20),
)

PHASE_COLOURS: Dict[str, Tuple[int, int, int]] = {
    "ready_stance": (150, 150, 150),
    "loading": (80, 180, 255),
    "ball_lift": (0, 210, 255),
    "jump": (0, 230, 120),
    "release": (60, 90, 255),
    "follow_through": (255, 170, 60),
    "landing": (200, 120, 255),
}
IDLE_COLOUR = (90, 90, 90)


def _panel(frame, lines: List[Tuple[str, Tuple[int, int, int], float]]) -> None:
    if not lines:
        return
    pad, line_h = 12, 26
    height = pad * 2 + line_h * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (330, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    y = pad + 18
    for text, colour, scale in lines:
        cv2.putText(frame, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    colour, 2, cv2.LINE_AA)
        y += line_h


def _timeline(frame, index: int, total: int, overlay_map) -> None:
    """A strip showing where the shots are, and where playback is in them."""
    h, w = frame.shape[:2]
    top, bar = h - 26, 14
    cv2.rectangle(frame, (0, top), (w, top + bar), (30, 30, 30), -1)
    for i, (_, phase, _) in overlay_map.items():
        if i >= total:
            continue
        x = int(w * i / max(total, 1))
        cv2.line(frame, (x, top), (x, top + bar),
                 PHASE_COLOURS.get(phase, IDLE_COLOUR), 1)
    cursor = int(w * index / max(total, 1))
    cv2.line(frame, (cursor, top - 4), (cursor, top + bar + 4), (255, 255, 255), 2)


def replay(
    video_path,
    landmarks: Sequence[Optional[List[Tuple[float, float]]]],
    overlay_map: Dict[int, Tuple[int, str, Optional[int]]],
    shot_count: int = 0,
    window_name: str = "Swichy - analysed",
    start_paused: bool = False,
) -> None:
    """Show the video with skeleton, phase and score drawn on every frame.

    Controls: SPACE pause/resume, LEFT/RIGHT step while paused, Q or ESC quit.
    """
    path = Path(video_path)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"Cannot open {path} for playback.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or len(landmarks)
    delay = max(1, int(1000.0 / fps))
    paused = start_paused
    index = 0

    print(f"\n  Playing back {path.name} with the analysis drawn on it.")
    print("  SPACE pause/resume · LEFT/RIGHT step when paused · Q quit\n")

    while True:
        if not paused or index == 0:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                break

        marks = landmarks[index] if index < len(landmarks) else None
        entry = overlay_map.get(index)
        phase = entry[1] if entry else None
        colour = PHASE_COLOURS.get(phase, IDLE_COLOUR) if phase else IDLE_COLOUR

        if marks:
            h, w = frame.shape[:2]
            pts = [(int(x * w), int(y * h)) for x, y in marks]
            for a, b in CONNECTIONS:
                if a < len(pts) and b < len(pts):
                    cv2.line(frame, pts[a], pts[b], colour, 2, cv2.LINE_AA)
            for i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28):
                if i < len(pts):
                    cv2.circle(frame, pts[i], 4, (255, 255, 255), -1, cv2.LINE_AA)

        lines = [(f"{index / fps:5.2f}s   frame {index}", (200, 200, 200), 0.6)]
        if entry:
            number, label, score = entry
            lines.append((f"SHOT {number}   {label.replace('_', ' ').upper()}",
                          colour, 0.7))
            if score is not None:
                lines.append((f"score {score}/100", (255, 255, 255), 0.6))
        else:
            lines.append(("no shot here", IDLE_COLOUR, 0.6))
        _panel(frame, lines)
        _timeline(frame, index, total, overlay_map)

        if shot_count:
            cv2.putText(frame, f"{shot_count} shot(s) found",
                        (frame.shape[1] - 250, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(window_name, frame)
        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        elif paused and key in (81, ord("a")):
            index = max(0, index - 2)
        elif paused and key in (83, ord("d")):
            pass
        if not paused or key in (81, 83, ord("a"), ord("d")):
            index += 1
        if index >= total:
            break

    cap.release()
    cv2.destroyAllWindows()

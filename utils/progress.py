"""
Progress reporting for a long analysis run.

WHY NOT FRAMES PER SECOND
-------------------------
The old display read:

    Source: 30.0 FPS | Processing: 12.3 FPS | 41% realtime

That number is true and it worries people for no reason. It is wall-clock
THROUGHPUT, not a health indicator: analysis reads every frame with a blocking
read and never skips, and timestamps come from the frame index and the source
frame rate, never from the clock. A run at 12 fps and a run at 30 fps analyse
exactly the same frames and produce exactly the same result -- one simply
finishes sooner. "41% realtime" reads as "59% of my video was ignored", which
is the opposite of what happened.

Progress is therefore reported as frames completed out of frames total, which
is unambiguous, plus an ETA, which is the thing someone waiting actually wants.
Throughput is still available for developers via `ProgressReporter.fps`.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Optional, TextIO


def _format_duration(seconds: float) -> str:
    if seconds < 1:
        return "<1s"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m {secs:02d}s"


@dataclass
class ProgressReporter:
    """Render a single, self-overwriting progress line.

    Silent when not attached to a terminal, so piping to a file or running
    under pytest produces no control characters.
    """

    total_frames: int
    label: str = "Analysing"
    stream: TextIO = field(default_factory=lambda: sys.stdout)
    bar_width: int = 24
    min_redraw_s: float = 0.1

    _started: float = field(default=0.0, init=False)
    _last_draw: float = field(default=0.0, init=False)
    _done: int = field(default=0, init=False)
    _active: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._started = time.perf_counter()
        self._active = bool(getattr(self.stream, "isatty", lambda: False)())

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._started

    @property
    def fps(self) -> float:
        """Throughput, for developers. Never shown to the player."""
        elapsed = self.elapsed_s
        return self._done / elapsed if elapsed > 0 else 0.0

    def advance(self, frames: int = 1, note: str = "") -> None:
        self._done += frames
        now = time.perf_counter()
        if now - self._last_draw < self.min_redraw_s and self._done < self.total_frames:
            return
        self._last_draw = now
        self._draw(note)

    def _draw(self, note: str = "") -> None:
        if not self._active:
            return
        total = max(self.total_frames, 1)
        fraction = min(self._done / total, 1.0)
        filled = int(round(self.bar_width * fraction))
        bar = "#" * filled + "." * (self.bar_width - filled)

        remaining = ""
        if 0.02 < fraction < 1.0:
            left = self.elapsed_s * (1.0 - fraction) / fraction
            remaining = f"  ~{_format_duration(left)} left"

        line = (
            f"\r  {self.label} [{bar}] {fraction * 100:3.0f}%  "
            f"frame {min(self._done, total)}/{total}{remaining}"
        )
        if note:
            line += f"  {note}"
        # Pad so a shorter line never leaves characters from a longer one.
        self.stream.write(line.ljust(96)[:96])
        self.stream.flush()

    def finish(self, message: Optional[str] = None) -> None:
        if not self._active:
            return
        self.stream.write("\r" + " " * 96 + "\r")
        if message:
            self.stream.write(f"  {message}\n")
        self.stream.flush()

"""Decide which frames are worth a pose inference, when nobody is on screen.

MediaPipe dominates the cost of reading a video, and on a long upload most of
what it is asked to do is confirm an empty court. This schedules those
inferences: dense while a person is anywhere in evidence, sparse once they have
been gone long enough to be believed gone, dense again the instant one returns.

It schedules POSE ONLY. It says nothing about decoding a frame and nothing
about the ball, and both of those omissions are the point -- see below.


WHY SKIPPING A PERSON-LESS FRAME CANNOT CHANGE THE POSE RESULT
--------------------------------------------------------------
`_offline_history` holds only frames where a pose was found, and is keyed by
timestamp rather than index (pipeline.py:1279). A frame with nobody in it never
enters the signal `segment` reads. Declining to pose it produces a history that
is identical, not merely similar.


WHY IT CANNOT CHANGE THE BALL EITHER
------------------------------------
The ball's state machine reads exactly three pose-derived inputs -- `wrist_xy`,
`ankle_y` and `pose_phase` -- and reads all three only up to and including
release: to begin an attempt, to move WAITING -> IN_HAND, and to confirm the
release (ball/shot_state_machine.py:212-265). After release it is velocity, rim
and physics. The ball goes on flying after the last pose phase precisely
because it has stopped needing one.

And on a frame with nobody in it those three inputs do not exist. Pose runs,
`extract_all_landmarks` returns None, and the FSM is called with
`wrist_xy=None, ankle_y=None, pose_phase=None` (pipeline.py:619-627). Not
running pose reaches the same call with the same three Nones. Byte for byte the
same update.

So the caller must keep decoding every frame and keep calling `process_frame`,
passing `None` where the detection result would go. What is saved is the
inference. What is preserved is the ball, and the parallel landmark array whose
absence once drew each skeleton one frame ahead of the player
(pipeline.py:597-605).


THE ONE WAY THIS CAN DO HARM, AND WHY IT IS OUT OF REACH
--------------------------------------------------------
A MISS -- skipping a frame that did have a person in it. The ball then loses
`wrist_xy` and `pose_phase` on a frame that had them, and those are what drive
IN_HAND and release confirmation.

But that damage is only possible BEFORE release, which is the one moment the
shooter is certainly in frame and moving; and this gate only opens after
`patience_s` of continuous absence. The window in which a miss would hurt is
the window in which the gate cannot be open. A miss after release costs
nothing, because nothing after release reads pose.

`patience_s` is therefore not a tuning knob for speed. It is the guarantee.
"""

from __future__ import annotations

from typing import Optional

# How long nobody may be seen before inferences are allowed to thin out.
#
# Long enough that a player who walks into frame is still sampled densely well
# before they could possibly shoot -- a shooting arc is about 2 s, and nobody
# arrives and releases inside 1.5 s of standing still. Shortening this trades
# that guarantee for inferences, and the trade is not worth making: the frames
# it buys are the cheapest ones in the clip, because an empty court is exactly
# where being late costs nothing.
DEFAULT_PATIENCE_S = 1.5

# How often to look, once nobody is believed to be there.
#
# This is the honest cost of the gate: the worst case for noticing someone who
# walks back in. At 0.4 s a returning player is dense-sampled again within
# roughly a tenth of a shot, and every downstream window has `FINE_PAD_S`
# (2.5 s) of reach behind its peak besides.
DEFAULT_IDLE_PERIOD_S = 0.4


class PresenceGate:
    """Which frames get a pose inference. Ask `should_pose`, then `observe`.

        gate = PresenceGate(fps)
        for i, frame in enumerate(video):
            result = None
            if gate.should_pose(i):
                result = detector.detect_video_frame(rgb, ts)
            pipeline.process_frame(result, w, h, ts, frame)   # ALWAYS
            gate.observe(i, bool(result and result.pose_landmarks))

    `observe` is separate from `should_pose` because a skipped frame teaches
    the gate nothing, and must not be allowed to. Reporting "no pose" for a
    frame that was never posed would let one absence justify the next, and the
    gate would latch shut on a court that had someone standing in it the whole
    time.
    """

    def __init__(
        self,
        fps: float,
        patience_s: float = DEFAULT_PATIENCE_S,
        idle_period_s: float = DEFAULT_IDLE_PERIOD_S,
        base_stride: int = 1,
    ) -> None:
        rate = fps if fps and fps > 0 else 30.0
        self._patience = max(1, int(round(rate * patience_s)))
        self._idle_step = max(1, int(round(rate * idle_period_s)))
        self._base = max(1, int(base_stride))
        # None, not 0, while nobody has been seen YET. A clip that opens on an
        # empty court is not "a person left just now"; it has never had one,
        # and the two deserve different answers -- the same distinction the
        # `or 0.0` ankle-baseline bug collapsed.
        self._last_seen: Optional[int] = None
        self._next: int = 0
        self.skipped = 0
        self.posed = 0

    def should_pose(self, frame_index: int) -> bool:
        return frame_index >= self._next

    def observe(self, frame_index: int, has_pose: bool) -> None:
        """Record the outcome of a frame that WAS posed. Never call it after a skip."""
        self.posed += 1
        if has_pose:
            self._last_seen = frame_index
            self._next = frame_index + self._base
            return
        # Absent, but not yet absent for long enough to be believed. A single
        # dropped detection in the middle of a shot is a MediaPipe miss, not an
        # empty court, and thinning out on the strength of one frame is how the
        # rest of that shot would be lost.
        gone = self._patience if self._last_seen is None else frame_index - self._last_seen
        step = self._idle_step if gone >= self._patience else self._base
        self._next = frame_index + step

    def skip(self, count: int = 1) -> None:
        """Count frames that went unposed, for the accounting only."""
        self.skipped += count

    @property
    def saved_fraction(self) -> float:
        total = self.posed + self.skipped
        return (self.skipped / total) if total else 0.0

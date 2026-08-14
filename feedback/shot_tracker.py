"""Find where each shot starts and ends, and score it when it completes.

WHY THIS IS ANCHORED ON THE RELEASE
-----------------------------------
The tracker used to work forwards: a posture change opened a candidate, and
the candidate then waited to see a release. Anything that stopped a candidate
being open at the right moment threw the shot away, and on continuous
practice footage that happened constantly.

The measurement that settled it: on a ten-shot practice video the state
machine detected a release at 17.43 s, correctly, and it was discarded --
there was no candidate open, because the previous shot's recovery had not
finished. Recovery after a shot ran three to five seconds; the gap between
shots was about 2.4 s. Every recovery swallowed the next shot. That is not a
threshold that needs tuning, it is a sequence that cannot fit, and four
attempts at retuning the constants each recovered one video by breaking
another.

So the dependency is inverted. The release is the event that defines a shot,
and it is detected independently of anything this class is doing. When one
arrives with no candidate open, the shot window is reconstructed BACKWARDS out
of the frame buffer. Whether the tracker was "ready" stops mattering.

WHY EVERY CANDIDATE IS BACK-FILLED
----------------------------------
Opening on a live signal always opens late -- the signal is the movement
already being underway. The knee dip that the loading rules score happens
before that. So every candidate, however it opened, is back-filled with the
frames leading up to it, and the dip is analysed whether or not the state
machine ever noticed it. That is what let the state machine drop from eight
states to four without losing any coaching detail.

All timing is in SECONDS, derived from frame timestamps. Sample material spans
12-30 fps and includes slow motion, so frame counts are not comparable.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Sequence

from ball.models import ShotOutcome
from feedback.generator import generate_coaching_tips
from feedback.models import ShotSummary
from feedback.performance_plan import build_shot_performance_plan
from feedback.phase_refiner import refine_phases
from feedback.scorer import score_shot
from phase_detection.phases import CORE_ACTIVE, CORE_ANCHOR, CORE_REST, PHASE_LABELS
from shots.classifier import AttemptEvidence, classify
from shots.elevation import shooting_event_ms, takeoff_elevation
from shots.types import RejectionReason, ShotType
from utils.config_loader import load_yaml
from utils.frame_buffer import FrameSnapshot

# Horizontal travel is measured over the shooting event only. Over the whole
# candidate it also counts the player walking out to collect the ball, and a
# stationary set shot was once reported as a driving layup because the hips had
# crossed 0.215 of the frame width between attempts.
_TRAVEL_STATES = frozenset({"rise", CORE_ANCHOR})

# Elevation is measured over the same window.
#
# REJECTED, on measurement: widening it to include the recovery. The argument
# was good -- a jump does not end when the ball does, the player is highest at
# or just after release and comes down through the follow-through, so cutting
# at the release samples the way up and misses the top. It was tried both
# unbounded and bounded to 0.8 s of plausible flight time. Neither recovered
# the under-measured jump shot it was aimed at (0.026 body heights, unchanged),
# and both turned a set shot in the ten-shot video into a jump shot. Sound
# reasoning, no measured gain, one measured loss.
_ELEVATION_STATES = _TRAVEL_STATES


@dataclass
class _Candidate:
    """A possible shot, accumulating evidence until confirmed or discarded."""

    started_ms: int
    entry_phase: str
    mid_start: bool
    frames: List[FrameSnapshot] = field(default_factory=list)
    reached_release: bool = False
    release_ms: Optional[int] = None
    body_finished: bool = False
    body_finished_ms: Optional[int] = None

    # Evidence
    wrist_min_y: float = float("inf")
    wrist_max_y: float = float("-inf")
    wrist_above_shoulder_max: float = float("-inf")
    # Image-space equivalents, in body heights. Used when MediaPipe's
    # visibility gate rejects the world wrist -- in which case `wrist_y` reads
    # a fabricated 0.0 and the world figures below are all zero, so a real
    # shot looks like a wrist that never moved.
    wrist_ratio_min: float = float("inf")
    wrist_ratio_max: float = float("-inf")
    wrist_ratio_shoulder_max: float = float("-inf")
    wrist_world_seen: bool = False
    ankle_image_seen: bool = False
    # Frame offset of the shooting event, when a caller decided the bounds
    # offline and therefore already knows where the hand peaked. None on the
    # live path, where the state machine's anchor is the only evidence.
    event_index: Optional[int] = None
    vertical_displacement_max: float = 0.0
    hip_x_min: float = float("inf")
    hip_x_max: float = float("-inf")
    last_ms: int = 0

    def observe(self, snapshot: FrameSnapshot) -> None:
        self.frames.append(snapshot)
        self.last_ms = max(self.last_ms, snapshot.timestamp_ms)
        f = snapshot.features
        if f is None:
            return

        if f.wrist_world_valid:
            self.wrist_world_seen = True
            self.wrist_min_y = min(self.wrist_min_y, f.wrist_y)
            self.wrist_max_y = max(self.wrist_max_y, f.wrist_y)
            self.wrist_above_shoulder_max = max(
                self.wrist_above_shoulder_max, f.wrist_y - f.shoulder_y
            )
        if f.ankle_image_y is not None:
            self.ankle_image_seen = True
        if f.wrist_height_ratio is not None:
            self.wrist_ratio_min = min(self.wrist_ratio_min, f.wrist_height_ratio)
            self.wrist_ratio_max = max(self.wrist_ratio_max, f.wrist_height_ratio)
        if f.wrist_above_shoulder_ratio is not None:
            self.wrist_ratio_shoulder_max = max(
                self.wrist_ratio_shoulder_max, f.wrist_above_shoulder_ratio
            )

        if snapshot.phase in _TRAVEL_STATES:
            self.hip_x_min = min(self.hip_x_min, f.hip_x_ratio)
            self.hip_x_max = max(self.hip_x_max, f.hip_x_ratio)
        if snapshot.phase in _ELEVATION_STATES:
            # body_rise_ratio comes from IMAGE space because world landmarks
            # are hip-centred and therefore blind to whole-body translation.
            self.vertical_displacement_max = max(
                self.vertical_displacement_max, f.body_rise_ratio
            )

    @property
    def duration_s(self) -> float:
        return max(0.0, (self.last_ms - self.started_ms) / 1000.0)

    @property
    def wrist_rise_m(self) -> float:
        if self.wrist_max_y == float("-inf") or self.wrist_min_y == float("inf"):
            return 0.0
        return self.wrist_max_y - self.wrist_min_y

    @property
    def wrist_rise_ratio(self) -> float:
        """How far the wrist travelled, in body heights (image space)."""
        if self.wrist_ratio_max == float("-inf") or self.wrist_ratio_min == float("inf"):
            return 0.0
        return self.wrist_ratio_max - self.wrist_ratio_min

    @property
    def horizontal_travel_m(self) -> float:
        """Horizontal hip travel across the frame, in image-normalised units."""
        if self.hip_x_max == float("-inf"):
            return 0.0
        return float(self.hip_x_max - self.hip_x_min)

    def takeoff_elevation(self) -> Optional[float]:
        """Elevation at the shooting event, against a stance baseline.

        Preferred over `vertical_displacement_max` because that figure is a
        running maximum of `body_rise_ratio` across the whole attempt, and
        `body_rise_ratio` is measured against a baseline that adapts frame by
        frame. Two failure modes follow from that, both observed:

          - the adaptive baseline drifts upward during a long flight, so the
            jump it is meant to measure shrinks as it happens (worst in slow
            motion, where the flight lasts longest);
          - a maximum over the whole attempt catches the player walking out to
            collect the ball, which reads as elevation because standing
            further away also lifts the feet in the image.

        Returns None when the stance cannot be observed, so the caller can
        decline to classify instead of guessing.
        """
        timestamps = [f.timestamp_ms for f in self.frames]
        ankles = [
            f.features.ankle_image_y if f.features is not None else None
            for f in self.frames
        ]
        heights = [
            f.features.body_pixel_height if f.features is not None else 0.0
            for f in self.frames
        ]
        wrists = [
            f.features.wrist_height_ratio if f.features is not None else None
            for f in self.frames
        ]
        event_ms = shooting_event_ms(timestamps, wrists)
        if event_ms is None:
            return None
        return takeoff_elevation(timestamps, ankles, heights, wrists, event_ms)

    def evidence(self) -> AttemptEvidence:
        elevation = self.takeoff_elevation()
        # Two different reasons the stance measurement can be missing, and only
        # one of them justifies refusing to classify.
        #
        # If the ankles were never seen in image space at all, there is nothing
        # to be strict about -- fall back to the legacy figure, which is what
        # every consumer had before this measurement existed.
        #
        # If the ankles WERE seen but no stance preceded the shot, we know the
        # recording opened mid-attempt. That is a real, identifiable gap in the
        # evidence, and guessing SET there mislabels a jump shot and then
        # scores it against the wrong mechanics.
        measured = elevation is not None or not self.ankle_image_seen
        return AttemptEvidence(
            vertical_displacement_m=(
                self.vertical_displacement_max if elevation is None else elevation
            ),
            vertical_displacement_measured=measured,
            horizontal_travel_m=self.horizontal_travel_m,
            wrist_rise_m=self.wrist_rise_m,
            wrist_above_shoulder_m=(
                self.wrist_above_shoulder_max
                if self.wrist_above_shoulder_max != float("-inf")
                else -1.0
            ),
            reached_release=self.reached_release,
            duration_s=self.duration_s,
        )


class ShotTracker:
    """Coordinate body-form capture with the terminal ball outcome."""

    def __init__(self):
        cfg = load_yaml("segmentation.yaml")
        confirm = cfg.get("confirmation", {}) or {}
        self._require_release = bool(confirm.get("require_release", True))
        self._min_wrist_rise_m = float(confirm.get("min_wrist_rise_m", 0.18))
        self._min_wrist_above_shoulder_m = float(
            confirm.get("min_wrist_above_shoulder_m", -0.12)
        )
        self._min_wrist_rise_ratio = float(confirm.get("min_wrist_rise_ratio", 0.20))
        self._min_wrist_above_shoulder_ratio = float(
            confirm.get("min_wrist_above_shoulder_ratio", -0.08)
        )
        self._min_duration_s = float(confirm.get("min_duration_s", 0.20))
        self._max_duration_s = float(confirm.get("max_duration_s", 12.0))
        self._post_release_s = float(cfg.get("post_release_s", 1.2))
        self._min_follow_through_s = float(cfg.get("min_follow_through_s", 0.25))
        self._candidate_timeout_s = float(cfg.get("candidate_timeout_s", 6.0))
        self._preroll_s = float(cfg.get("preroll_s", 1.5))
        self._max_lookback_s = float(cfg.get("max_lookback_s", 4.0))

        self.shot_count = 0
        self.last_summary: Optional[ShotSummary] = None
        self.last_score: Optional[int] = None
        self.discarded_candidates = 0

        self._prev_phase = CORE_REST
        self._candidate: Optional[_Candidate] = None
        self._summary_display_frames = 0
        self._first_shot_pending = True
        # Frames up to here already belong to a finished attempt and may not be
        # pulled into a new one.
        self._consumed_until_ms: Optional[int] = None

        self._history: Optional[Sequence[FrameSnapshot]] = None
        self._analyzer = None

        self._require_ball_outcome = False
        self._body_grace_ms = 500
        self._ball_outcome: Optional[ShotOutcome] = None
        self._ball_outcome_timestamp_ms: Optional[int] = None

    # ------------------------------------------------------------------ api
    def attach_history(self, frame_buffer) -> None:
        """Give the tracker the recent past, so it can open backwards.

        Without this the release anchor still works, but the shot window
        starts at the release instead of at the dip -- everything before the
        anchor would be missing and only the finish could be scored.
        """
        self._history = frame_buffer

    def attach_analyzer(self, engine) -> None:
        """Supply the rule engine used to re-score refined phases.

        Rules are evaluated live against the four detector states, which is
        enough for the on-screen display but not for the report: the report
        talks about seven coaching phases that are only worked out once the
        shot is complete. The engine is handed in rather than constructed here
        so it carries the same player profile and the same config as the live
        pass, instead of a second engine that could silently disagree.
        """
        self._analyzer = engine

    @property
    def shot_in_progress(self) -> bool:
        return self._candidate is not None

    @property
    def capture_in_progress(self) -> bool:
        return self._candidate is not None and not self._candidate.body_finished

    @property
    def show_shot_summary(self) -> bool:
        return self._summary_display_frames > 0 and self.last_summary is not None

    @property
    def capture_warning(self) -> Optional[str]:
        c = self._candidate
        if c is not None and c.mid_start:
            label = PHASE_LABELS.get(c.entry_phase, c.entry_phase)
            return f"Recording started at {label} — earlier phases not on camera"
        return None

    def configure_ball_outcome(self, required: bool, body_grace_ms: int = 500) -> None:
        self._require_ball_outcome = bool(required)
        self._body_grace_ms = max(0, int(body_grace_ms))

    def begin_summary_display(self, frames: int) -> None:
        self._summary_display_frames = frames

    def reset(self) -> None:
        required, grace = self._require_ball_outcome, self._body_grace_ms
        history, analyzer = self._history, self._analyzer
        self.__init__()
        self.configure_ball_outcome(required, grace)
        self._history, self._analyzer = history, analyzer

    # -------------------------------------------------------------- updates
    def update(
        self,
        phase: str,
        snapshot: FrameSnapshot,
        ball_outcome: Optional[ShotOutcome] = None,
    ) -> Optional[ShotSummary]:
        now_ms = snapshot.timestamp_ms
        completed: Optional[ShotSummary] = None

        if self._candidate is None:
            self._maybe_open(phase, now_ms)
        elif phase == CORE_ANCHOR and not self._candidate.reached_release:
            self._candidate.reached_release = True
            self._candidate.release_ms = now_ms

        if self._candidate is not None and not self._candidate.body_finished:
            self._candidate.observe(snapshot)
            if self._body_has_finished(phase):
                self._candidate.body_finished = True
                self._candidate.body_finished_ms = now_ms

        self._accept_ball_outcome(ball_outcome, now_ms)

        if self._candidate is not None:
            if (
                self._candidate.duration_s > self._candidate_timeout_s
                and not self._candidate.reached_release
            ):
                # No release ever arrived. This is the toe-jiggle path.
                self._discard()
            elif self._release_window_expired(now_ms, snapshot):
                # Released, and the follow-through has had its time. Close on
                # the clock rather than waiting for a state transition that may
                # never arrive, because an open candidate absorbs every later
                # attempt into itself.
                self._candidate.body_finished = True
                completed = self._finalize(now_ms)
            elif self._ready_to_finalize(now_ms):
                completed = self._finalize(now_ms)

        self._prev_phase = phase
        self._advance_summary_display(completed)
        return completed

    def update_ball_outcome(
        self,
        outcome: Optional[ShotOutcome],
        timestamp_ms: int,
    ) -> Optional[ShotSummary]:
        """Continue/finalize an active attempt when pose is absent this frame."""
        if self._candidate is None:
            self._advance_summary_display(None)
            return None

        self._accept_ball_outcome(outcome, timestamp_ms)
        completed = None
        if self._ready_to_finalize(timestamp_ms):
            completed = self._finalize(timestamp_ms)
        self._advance_summary_display(completed)
        return completed

    def finalize_in_progress(self, min_frames: int = 12) -> Optional[ShotSummary]:
        """Close an attempt still open when the video or session ends."""
        c = self._candidate
        if c is None or len(c.frames) < min_frames:
            self._candidate = None
            return None
        if self._require_ball_outcome and self._ball_outcome is None:
            self._ball_outcome = self._unknown_outcome(
                c.last_ms, "Session ended before ball outcome was resolved"
            )
        summary = self._finalize(c.last_ms, force=True)
        if summary is not None:
            self._summary_display_frames = 90
        return summary

    # ------------------------------------------------------------- opening
    def _maybe_open(self, phase: str, now_ms: int) -> None:
        """Start a candidate, live or retroactively.

        The anchor path is checked first and deliberately obeys no guard other
        than "these frames are not already spoken for". A release is the
        strongest evidence a shot exists, so anything that would suppress it
        costs a real shot.
        """
        if phase == CORE_ANCHOR and self._prev_phase != CORE_ANCHOR:
            if self._consumed_until_ms is None or now_ms > self._consumed_until_ms:
                self._open_at_anchor(now_ms)
                return

        if self._is_start_transition(phase) or (
            self._first_shot_pending and phase in CORE_ACTIVE
        ):
            mid = self._first_shot_pending and not self._is_start_transition(phase)
            self._open(phase, now_ms, mid_start=mid)

    def _open(self, phase: str, now_ms: int, mid_start: bool) -> None:
        self._candidate = _Candidate(
            started_ms=now_ms, entry_phase=phase, mid_start=mid_start
        )
        self._first_shot_pending = False
        self._ball_outcome = None
        self._ball_outcome_timestamp_ms = None
        self._backfill(now_ms)

    def _open_at_anchor(self, now_ms: int) -> None:
        """Reconstruct the shot window backwards from a release."""
        self._open(CORE_ANCHOR, now_ms, mid_start=False)
        c = self._candidate
        if c is not None:
            c.reached_release = True
            c.release_ms = now_ms
            if c.frames:
                c.started_ms = c.frames[0].timestamp_ms
                # The window was rebuilt from history, so nothing is missing
                # from the start even though the candidate opened at the
                # release. `mid_start` would tell the player their run-up was
                # off camera, which would be untrue.
                c.mid_start = False
                c.entry_phase = c.frames[0].phase

    def _backfill(self, now_ms: int) -> None:
        """Pull the frames leading up to this moment into the candidate.

        Bounded three ways: never past a previous attempt's frames, never
        further back than `max_lookback_s`, and never past the point where the
        player was last at rest for longer than the pre-roll. The last bound is
        what stops a candidate from swallowing the walk out to collect the ball.
        """
        c = self._candidate
        if c is None or self._history is None:
            return
        frames = list(getattr(self._history, "frames", self._history) or [])
        if not frames:
            return

        floor_ms = now_ms - int(self._max_lookback_s * 1000)
        if self._consumed_until_ms is not None:
            floor_ms = max(floor_ms, self._consumed_until_ms + 1)

        window = [
            s
            for s in frames
            if floor_ms <= s.timestamp_ms < now_ms
        ]
        if not window:
            return

        # Walk back from the newest frame and stop once the player has been at
        # rest for longer than the pre-roll. Everything before that belongs to
        # the gap between shots, not to this shot.
        kept: List[FrameSnapshot] = []
        rest_run_ms = 0
        prev_ms = None
        for s in reversed(window):
            if prev_ms is not None:
                gap = max(0, prev_ms - s.timestamp_ms)
                rest_run_ms = rest_run_ms + gap if s.phase == CORE_REST else 0
                if rest_run_ms > self._preroll_s * 1000:
                    break
            kept.append(s)
            prev_ms = s.timestamp_ms

        for s in reversed(kept):
            c.observe(s)
        if c.frames:
            c.started_ms = c.frames[0].timestamp_ms

    # ----------------------------------------------------------- internals
    def _is_start_transition(self, phase: str) -> bool:
        return self._prev_phase == CORE_REST and phase == "rise"

    def _release_window_expired(self, now_ms: int, snapshot=None) -> bool:
        """Is the shooting motion over, now that the ball has gone?

        Two ways to answer, and the order matters.

        FIRST, the physical one: the shooting hand comes back down. After
        release the arm holds its finish and then drops, so a wrist that has
        returned below the shoulder means this attempt is done. That adapts by
        itself -- a jump shot with a long hang and a set shot fired every two
        seconds both end when the hand comes down, at whatever pace they run.

        SECOND, a clock, purely as a backstop for when the wrist cannot be
        seen well enough to observe the descent.

        Closing PROMPTLY matters much less than it used to. A candidate that
        overstays no longer blocks the next shot, because the next release
        opens its own window regardless. So this can now favour the physical
        signal without the clock having to defend against merged attempts.
        """
        c = self._candidate
        if c is None or c.release_ms is None or c.body_finished:
            return False

        elapsed_s = (now_ms - c.release_ms) / 1000.0
        # A brief floor so the descent is not detected on the release frame
        # itself, where the wrist may still be reported below the shoulder.
        if elapsed_s >= self._min_follow_through_s and snapshot is not None:
            f = snapshot.features
            if f is not None and self._wrist_has_dropped(f):
                return True
        return elapsed_s > self._post_release_s

    @staticmethod
    def _wrist_has_dropped(f) -> bool:
        """Has the shooting hand come back down below the shoulder?"""
        if f.wrist_above_shoulder_ratio is not None:
            return f.wrist_above_shoulder_ratio < 0.0
        if f.wrist_world_valid:
            return f.wrist_y < f.shoulder_y
        return False

    def _body_has_finished(self, phase: str) -> bool:
        """Has the shooting motion returned to rest?

        Returning to rest from ANY state ends the capture, provided the ball
        was actually released. Without the release check a candidate that
        opened on a stance adjustment would close immediately and be scored as
        a shot.
        """
        c = self._candidate
        if c is None or phase != CORE_REST:
            return False
        return c.reached_release

    def _discard(self) -> None:
        self.discarded_candidates += 1
        self._candidate = None
        self._ball_outcome = None
        self._ball_outcome_timestamp_ms = None

    def _is_credible(self, c: _Candidate, require_landing: bool = True) -> bool:
        """Does this candidate show a real shooting event?"""
        if self._require_release and not c.reached_release:
            return False
        if c.duration_s < self._min_duration_s:
            return False
        if c.duration_s > self._max_duration_s:
            return False
        return self._wrist_travelled(c)

    def _wrist_travelled(self, c: _Candidate) -> bool:
        """Did the shooting hand actually make a shooting motion?

        Measured in world metres when the wrist was genuinely observed, and in
        body heights otherwise. Without the fallback, footage where MediaPipe's
        visibility gate rejects the wrist -- filming from behind does it
        routinely -- reports zero wrist travel and every real shot is thrown
        away as preparation.
        """
        by_world = c.wrist_world_seen and (
            c.wrist_rise_m >= self._min_wrist_rise_m
            and c.wrist_above_shoulder_max >= self._min_wrist_above_shoulder_m
        )
        by_image = c.wrist_ratio_max != float("-inf") and (
            c.wrist_rise_ratio >= self._min_wrist_rise_ratio
            and c.wrist_ratio_shoulder_max >= self._min_wrist_above_shoulder_ratio
        )
        # Either measurement is sufficient. They disagree only when one of them
        # is blind, never when the wrist genuinely stayed still: a hand that
        # did not move produces a small number in both spaces.
        return by_world or by_image

    def _accept_ball_outcome(self, outcome, timestamp_ms: int) -> None:
        if self._candidate is None or outcome is None or self._ball_outcome is not None:
            return
        self._ball_outcome = outcome
        self._ball_outcome_timestamp_ms = (
            outcome.outcome_timestamp_ms
            if outcome.outcome_timestamp_ms is not None
            else timestamp_ms
        )

    def _ready_to_finalize(self, timestamp_ms: int) -> bool:
        c = self._candidate
        if c is None:
            return False
        if not self._require_ball_outcome:
            return c.body_finished
        if self._ball_outcome is None:
            return False
        if c.body_finished:
            return True
        if self._ball_outcome_timestamp_ms is None:
            return False
        return timestamp_ms - self._ball_outcome_timestamp_ms >= self._body_grace_ms

    # ---------------------------------------------------------- finalising
    def _refined_frames(
        self, c: _Candidate, shot_type=None
    ) -> List[FrameSnapshot]:
        """Relabel the captured frames with coaching phases and re-score them.

        The live pass evaluated rules against the four detector states, so a
        rule that only runs in `loading` or `landing` never fired. Re-running
        the engine against the refined labels is what puts those rules back --
        and it does so with the whole shot in view, which is strictly more
        information than the live pass had.

        `shot_type` is the settled classification. It is what lets a rule
        declare `shot_types:` and be skipped for a shot that never performs the
        mechanic -- a set shot has no flight to assess, and scoring one against
        a jump-shot rule would invent a verdict rather than withhold it.
        """
        if not c.frames:
            return []
        labels = refine_phases(c.frames, c.event_index)
        type_name = getattr(shot_type, "value", shot_type)
        out: List[FrameSnapshot] = []
        for snap, label in zip(c.frames, labels):
            analysis = snap.analysis
            if self._analyzer is not None:
                analysis = self._analyzer.evaluate(
                    label, snap.angles, snap.features, snap.shooting_side,
                    shot_type=type_name,
                )
            out.append(replace(snap, phase=label, analysis=analysis))
        return out

    def finalize_window(
        self, frames: Sequence[FrameSnapshot], peak_ms: Optional[int] = None
    ) -> Optional[ShotSummary]:
        """Score one attempt whose bounds were decided offline.

        The live path discovers an attempt while it is happening and must judge
        the evidence as it arrives. Here the bounds already came from
        `shots.segmenter`, which saw the whole video, so the questions the live
        path asks -- has the wrist travelled far enough, did a release happen,
        has the body finished -- are already answered by the window existing at
        all. What remains is the part that was never the problem: relabel the
        phases, classify, score.

        Landing is not required. Offline bounds are cut where the hand comes
        back down, so an attempt whose landing falls outside the window is
        bounded correctly, not truncated.
        """
        usable = [f for f in frames if f is not None]
        if not usable:
            return None

        candidate = _Candidate(
            started_ms=usable[0].timestamp_ms,
            entry_phase=usable[0].phase,
            mid_start=False,
        )
        for snapshot in usable:
            candidate.observe(snapshot)
        # The peak IS the shooting event: the segmenter found this window
        # because the hand rose and came back down around it.
        candidate.reached_release = True
        candidate.release_ms = peak_ms if peak_ms is not None else candidate.last_ms
        if peak_ms is not None:
            candidate.event_index = min(
                range(len(usable)),
                key=lambda i: abs(usable[i].timestamp_ms - peak_ms),
            )
        candidate.body_finished = True
        candidate.body_finished_ms = candidate.last_ms

        previous, self._candidate = self._candidate, candidate
        try:
            return self._finalize(
                candidate.last_ms, force=True, trust_bounds=True
            )
        finally:
            self._candidate = previous

    def _finalize(
        self, now_ms: int, force: bool = False, trust_bounds: bool = False
    ) -> Optional[ShotSummary]:
        c = self._candidate
        if c is None:
            return None

        # An attempt can be finalised by the ball outcome on a frame where no
        # pose was observed, so the candidate's own clock may lag. The attempt
        # genuinely spans up to this moment.
        c.last_ms = max(c.last_ms, now_ms)

        # `force` (the video ended mid-attempt) relaxes only the requirement
        # that the body was seen to land. It must NOT relax the evidence that
        # a shot happened at all -- a clip that stops during a stray movement
        # would otherwise be closed out as a scored shot.
        # The credibility check asks whether a shot happened at all. Offline
        # bounds have already answered that: the window exists because the hand
        # rose more than half a body height above everything around it, which
        # is stronger evidence than any of the live thresholds below. Re-asking
        # discards real shots -- measured, it threw away the only attempt in
        # video_07 and in video_01, both of which the segmenter had located
        # correctly.
        if not trust_bounds and not self._is_credible(c, require_landing=not force):
            self._discard()
            return None

        classification = classify(c.evidence())
        # The classification is settled BEFORE the rules are re-run, so a rule
        # scoped to one kind of shot can be evaluated against the shot it
        # actually was rather than against a guess made while it was happening.
        frames = self._refined_frames(c, classification.shot_type)

        self.shot_count += 1
        if not classification.is_implemented:
            # Recognised, but we have no model for it. Produce a rejection,
            # never a score computed with the wrong biomechanics.
            summary = ShotSummary(
                shot_number=self.shot_count,
                score=None,
                passed_count=0,
                total_count=0,
                phases_seen=_phases_seen(frames),
                entry_phase=c.entry_phase,
                shot_type=classification.shot_type,
                classification=classification,
                rejection=classification.rejection
                or RejectionReason.SHOT_TYPE_NOT_SUPPORTED_YET,
            )
        else:
            summary = score_shot(
                frames,
                self.shot_count,
                started_mid_phase=c.mid_start,
                ended_early=not c.body_finished,
                entry_phase=frames[0].phase if frames else c.entry_phase,
            )
            summary.shot_type = classification.shot_type
            summary.classification = classification
            summary.outcome = self._ball_outcome
            summary.coaching_tips = generate_coaching_tips(summary)
            summary = build_shot_performance_plan(summary)

        # Real frame timestamps, on both the scored and the rejected path.
        summary.start_timestamp_ms = c.started_ms
        summary.end_timestamp_ms = c.last_ms

        self.last_summary = summary
        self.last_score = summary.score
        self._consumed_until_ms = max(c.last_ms, now_ms)
        self._candidate = None
        self._ball_outcome = None
        self._ball_outcome_timestamp_ms = None
        return summary

    @staticmethod
    def _unknown_outcome(timestamp_ms: int, evidence: str) -> ShotOutcome:
        return ShotOutcome(
            result="unknown",
            confidence=0.0,
            outcome_timestamp_ms=timestamp_ms,
            evidence=[evidence],
        )

    def _advance_summary_display(self, completed: Optional[ShotSummary]) -> None:
        if completed is not None:
            self._summary_display_frames = 90
        elif self._summary_display_frames > 0:
            self._summary_display_frames -= 1


def _phases_seen(frames: Sequence[FrameSnapshot]) -> List[str]:
    seen: List[str] = []
    for f in frames:
        if f.phase not in seen:
            seen.append(f.phase)
    return seen

"""Shared evidence labels for independent pose and ball release clocks."""

from __future__ import annotations

from typing import Optional


def release_timing_fields(
    pose_release_timestamp_ms: Optional[int],
    ball_release_timestamp_ms: Optional[int],
    disagreement_limit_ms: int,
) -> dict:
    """Return timing evidence without choosing one detector as the winner.

    The configured limit is a quality boundary, not a replacement timestamp:
    pose still anchors coaching phases and ball still anchors flight physics.
    """
    limit_ms = max(0, int(disagreement_limit_ms))
    signed_disagreement = None
    if (
        pose_release_timestamp_ms is not None
        and ball_release_timestamp_ms is not None
    ):
        signed_disagreement = (
            int(ball_release_timestamp_ms)
            - int(pose_release_timestamp_ms)
        )

    absolute_disagreement = (
        abs(signed_disagreement)
        if signed_disagreement is not None
        else None
    )
    if absolute_disagreement is None:
        timing_status = "unavailable"
        alignment_confidence = "unassessed"
    elif absolute_disagreement <= limit_ms:
        timing_status = "aligned"
        alignment_confidence = "normal"
    else:
        timing_status = "disagreed"
        alignment_confidence = "low"

    return {
        "pose_release_timestamp_ms": pose_release_timestamp_ms,
        "ball_release_timestamp_ms": ball_release_timestamp_ms,
        "release_disagreement_ms": absolute_disagreement,
        "ball_minus_pose_release_ms": signed_disagreement,
        "release_disagreement_limit_ms": limit_ms,
        "release_timing_status": timing_status,
        "release_alignment_confidence": alignment_confidence,
    }

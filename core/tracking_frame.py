"""Normalized tracking sample shared by input and output adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass(slots=True)
class TrackingFrame:
    """Technology-independent tracking sample.

    Position values intentionally retain a unit label because the current POC
    mixes image-space trackers (pixels) and external 6-DOF trackers whose
    translation unit is defined by the source application. A future calibrated
    pose layer can standardize all position values to meters without changing
    output-adapter APIs.
    """

    timestamp: float = field(default_factory=time.time)
    source: str = ""
    target_id: int | None = None
    tracking_valid: bool = False
    confidence: float = 0.0

    x: float | None = None
    y: float | None = None
    z: float | None = None
    position_unit: str = "unknown"

    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None

    @classmethod
    def invalid(
        cls,
        source: str,
        target_id: int | None = None,
        position_unit: str = "unknown",
    ) -> "TrackingFrame":
        """Create an explicit no-tracking sample."""
        return cls(
            source=source,
            target_id=target_id,
            tracking_valid=False,
            confidence=0.0,
            position_unit=position_unit,
        )

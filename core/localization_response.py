"""Experiment-facing localization response model."""

from __future__ import annotations

from dataclasses import dataclass, field
import time


@dataclass(slots=True)
class LocalizationResponse:
    """Technology-independent localization response.

    Azimuth is expressed in degrees relative to the calibrated speaker-array
    reference. By convention, 0 degrees is the calibrated forward/center
    direction. Positive/negative sign follows the configured resolver sign.
    """

    timestamp: float = field(default_factory=time.time)
    source: str = ""
    response_method: str = "head_orientation"
    response_valid: bool = False
    confidence: float = 0.0

    response_azimuth_deg: float | None = None
    response_elevation_deg: float | None = None
    target_id: int | None = None

    @classmethod
    def invalid(
        cls,
        source: str,
        response_method: str = "head_orientation",
        target_id: int | None = None,
    ) -> "LocalizationResponse":
        return cls(
            source=source,
            response_method=response_method,
            response_valid=False,
            confidence=0.0,
            target_id=target_id,
        )

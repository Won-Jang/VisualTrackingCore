"""Convert raw tracking pose into speaker-array-relative localization responses."""

from __future__ import annotations

from core.localization_response import LocalizationResponse
from core.tracking_frame import TrackingFrame


def wrap_degrees(angle: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (float(angle) + 180.0) % 360.0 - 180.0


class ResponseResolver:
    """Resolve tracking orientation into a calibrated localization azimuth.

    The resolver intentionally does not know about stimulus speakers or trial
    correctness. It only converts the current tracking measurement into a
    response direction relative to a user-established 0-degree reference.
    """

    def __init__(
        self,
        response_method: str = "head_orientation",
        orientation_axis: str = "yaw",
        sign: float = 1.0,
        orientation_label: str | None = None,
    ):
        if orientation_axis not in {"yaw", "pitch", "roll"}:
            raise ValueError("orientation_axis must be yaw, pitch, or roll")
        if sign == 0:
            raise ValueError("sign must be non-zero")

        self.response_method = response_method
        self.orientation_axis = orientation_axis
        self.sign = 1.0 if sign > 0 else -1.0
        self.orientation_label = orientation_label or orientation_axis.capitalize()
        self.reference_deg: float | None = None

    @property
    def calibrated(self) -> bool:
        return self.reference_deg is not None

    def _orientation_value(self, frame: TrackingFrame) -> float | None:
        return getattr(frame, f"{self.orientation_axis}_deg")

    def calibrate(self, frame: TrackingFrame) -> float:
        """Set the current tracker orientation as the speaker-array 0-degree reference."""
        if not frame.tracking_valid:
            raise ValueError("Cannot calibrate while tracking is invalid.")

        value = self._orientation_value(frame)
        if value is None:
            raise ValueError(
                f"Current adapter does not provide {self.orientation_axis} orientation."
            )

        self.reference_deg = float(value)
        return self.reference_deg

    def clear_reference(self) -> None:
        self.reference_deg = None

    def resolve(self, frame: TrackingFrame) -> LocalizationResponse:
        if not frame.tracking_valid or not self.calibrated:
            return LocalizationResponse.invalid(
                source=frame.source,
                response_method=self.response_method,
                target_id=frame.target_id,
            )

        value = self._orientation_value(frame)
        if value is None:
            return LocalizationResponse.invalid(
                source=frame.source,
                response_method=self.response_method,
                target_id=frame.target_id,
            )

        azimuth = self.sign * wrap_degrees(float(value) - self.reference_deg)

        return LocalizationResponse(
            timestamp=frame.timestamp,
            source=frame.source,
            response_method=self.response_method,
            response_valid=True,
            confidence=frame.confidence,
            response_azimuth_deg=azimuth,
            response_elevation_deg=None,
            target_id=frame.target_id,
        )

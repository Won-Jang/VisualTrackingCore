"""Abstract interface that every VisualTrackingCore input adapter implements."""

from abc import ABC, abstractmethod

from core.tracking_frame import TrackingFrame
from tracking_types import TrackingResult


class TrackingAdapter(ABC):
    """Common interface for all tracking solutions.

    ``detect`` is retained as the V0.6-compatible input contract. The new core
    converts each TrackingResult into a technology-independent TrackingFrame
    before any output transport sees it.
    """

    name = "Base Adapter"
    uses_camera = True
    position_unit = "px"

    def __init__(self, target_id: int | None = 0):
        self.target_id = target_id

    @abstractmethod
    def detect(self, frame) -> list[TrackingResult]:
        """Analyze one input frame and return zero or more results."""
        raise NotImplementedError

    def to_tracking_frame(self, result: TrackingResult) -> TrackingFrame:
        """Convert the legacy/UI result to the normalized core data model."""
        source_x = result.source_x
        source_y = result.source_y
        source_z = result.source_z

        if source_x is not None or source_y is not None or source_z is not None:
            x, y, z = source_x, source_y, source_z
            unit = self.position_unit
        else:
            x, y, z = result.x_px, result.y_px, None
            unit = "px"

        roll = result.roll_deg
        if roll is None:
            # Marker adapters currently report in-plane orientation as angle_deg.
            roll = result.angle_deg

        return TrackingFrame(
            source=self.name,
            target_id=result.marker_id,
            tracking_valid=True,
            confidence=1.0,
            x=x,
            y=y,
            z=z,
            position_unit=unit,
            yaw_deg=result.yaw_deg,
            pitch_deg=result.pitch_deg,
            roll_deg=roll,
        )

    def invalid_tracking_frame(self) -> TrackingFrame:
        """Return an explicit invalid sample when the target is not detected."""
        return TrackingFrame.invalid(
            source=self.name,
            target_id=self.target_id,
            position_unit=self.position_unit if not self.uses_camera else "px",
        )

    def close(self) -> None:
        """Optional cleanup hook for adapters that need it."""
        pass

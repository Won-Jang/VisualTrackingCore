"""Abstract interface that every VisualTrackingCore adapter implements."""

from abc import ABC, abstractmethod
from tracking_types import TrackingResult


class TrackingAdapter(ABC):
    """Common interface for all tracking solutions."""

    name = "Base Adapter"
    uses_camera = True

    def __init__(self, target_id: int | None = 0):
        self.target_id = target_id

    @abstractmethod
    def detect(self, frame) -> list[TrackingResult]:
        """Analyze one BGR OpenCV frame and return zero or more results."""
        raise NotImplementedError

    def close(self) -> None:
        """Optional cleanup hook for adapters that need it."""
        pass

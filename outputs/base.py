"""Abstract interface for VisualTrackingCore output transports."""

from abc import ABC, abstractmethod

from core.tracking_frame import TrackingFrame


class OutputAdapter(ABC):
    name = "Base Output"

    @abstractmethod
    def start(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def send(self, frame: TrackingFrame) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        pass

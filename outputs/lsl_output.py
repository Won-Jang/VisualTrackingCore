"""Lab Streaming Layer output for normalized VisualTrackingCore samples."""

from __future__ import annotations

import math

from core.tracking_frame import TrackingFrame
from outputs.base import OutputAdapter


class LSLOutputAdapter(OutputAdapter):
    name = "LSL"

    CHANNELS = (
        ("x", "position"),
        ("y", "position"),
        ("z", "position"),
        ("yaw", "degrees"),
        ("pitch", "degrees"),
        ("roll", "degrees"),
        ("confidence", "normalized"),
        ("tracking_valid", "boolean"),
        ("target_id", "id"),
    )

    def __init__(
        self,
        stream_name: str = "VisualTrackingCore_Tracking",
        stream_type: str = "TrackingPose",
        source_name: str = "",
        position_unit: str = "unknown",
    ):
        self.stream_name = stream_name.strip() or "VisualTrackingCore_Tracking"
        self.stream_type = stream_type
        self.source_name = source_name
        self.position_unit = position_unit
        self.outlet = None
        self._local_clock = None

    def start(self) -> None:
        try:
            from pylsl import StreamInfo, StreamOutlet, cf_float32, local_clock
        except ImportError as exc:
            raise RuntimeError(
                "LSL output requires the 'pylsl' package. Install project "
                "requirements with: python -m pip install -r requirements.txt"
            ) from exc

        info = StreamInfo(
            self.stream_name,
            self.stream_type,
            len(self.CHANNELS),
            0.0,
            cf_float32,
            f"VisualTrackingCore-{self.source_name or 'tracking'}",
        )

        desc = info.desc()
        desc.append_child_value("application", "VisualTrackingCore")
        desc.append_child_value("source_adapter", self.source_name)
        desc.append_child_value("position_unit", self.position_unit)
        desc.append_child_value("data_role", "raw_normalized_tracking")

        channels = desc.append_child("channels")
        for label, unit in self.CHANNELS:
            channel = channels.append_child("channel")
            channel.append_child_value("label", label)
            if label in ("x", "y", "z"):
                channel.append_child_value("unit", self.position_unit)
            else:
                channel.append_child_value("unit", unit)

        self.outlet = StreamOutlet(info)
        self._local_clock = local_clock

    @staticmethod
    def _number(value: float | None) -> float:
        return math.nan if value is None else float(value)

    def send(self, frame: TrackingFrame) -> None:
        if self.outlet is None:
            return

        sample = [
            self._number(frame.x),
            self._number(frame.y),
            self._number(frame.z),
            self._number(frame.yaw_deg),
            self._number(frame.pitch_deg),
            self._number(frame.roll_deg),
            float(frame.confidence),
            1.0 if frame.tracking_valid else 0.0,
            float(frame.target_id) if frame.target_id is not None else -1.0,
        ]

        # LSL timestamps must use the LSL local clock domain, not Unix time.
        self.outlet.push_sample(sample, timestamp=self._local_clock())

    def stop(self) -> None:
        self.outlet = None
        self._local_clock = None

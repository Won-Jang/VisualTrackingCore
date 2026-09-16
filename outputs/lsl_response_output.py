"""LSL output for calibrated localization responses."""

from __future__ import annotations

import math

from core.localization_response import LocalizationResponse
from outputs.base import OutputAdapter


class LSLResponseOutputAdapter(OutputAdapter):
    name = "LSL Response"

    CHANNELS = (
        ("response_azimuth", "degrees"),
        ("response_elevation", "degrees"),
        ("confidence", "normalized"),
        ("response_valid", "boolean"),
        ("target_id", "id"),
    )

    def __init__(
        self,
        stream_name: str = "VisualTrackingCore_Response",
        source_name: str = "",
        response_method: str = "head_orientation",
        metadata: dict | None = None,
    ):
        self.stream_name = stream_name.strip() or "VisualTrackingCore_Response"
        self.source_name = source_name
        self.response_method = response_method
        self.metadata = metadata or {}
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
            "LocalizationResponse",
            len(self.CHANNELS),
            0.0,
            cf_float32,
            f"VisualTrackingCore-{self.source_name or 'response'}-response",
        )

        desc = info.desc()
        desc.append_child_value("application", "VisualTrackingCore")
        desc.append_child_value("source_adapter", self.source_name)
        desc.append_child_value("response_method", self.response_method)
        desc.append_child_value("coordinate_reference", "calibrated_speaker_array")
        desc.append_child_value("azimuth_zero", "calibrated_forward_center")
        for key, value in self.metadata.items():
            desc.append_child_value(key, str(value))

        channels = desc.append_child("channels")
        for label, unit in self.CHANNELS:
            channel = channels.append_child("channel")
            channel.append_child_value("label", label)
            channel.append_child_value("unit", unit)

        self.outlet = StreamOutlet(info)
        self._local_clock = local_clock

    @staticmethod
    def _number(value: float | None) -> float:
        return math.nan if value is None else float(value)

    def send(self, response: LocalizationResponse) -> None:
        if self.outlet is None:
            return

        sample = [
            self._number(response.response_azimuth_deg),
            self._number(response.response_elevation_deg),
            float(response.confidence),
            1.0 if response.response_valid else 0.0,
            float(response.target_id) if response.target_id is not None else -1.0,
        ]
        self.outlet.push_sample(sample, timestamp=self._local_clock())

    def stop(self) -> None:
        self.outlet = None
        self._local_clock = None

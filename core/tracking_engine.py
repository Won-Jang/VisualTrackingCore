"""Routes normalized tracking frames and localization responses to outputs."""

from __future__ import annotations

from collections.abc import Iterable

from core.localization_response import LocalizationResponse
from core.response_resolver import ResponseResolver
from core.tracking_frame import TrackingFrame


class TrackingEngine:
    """Routing core between an input adapter, response resolver, and outputs."""

    def __init__(self):
        self.input_adapter = None
        self.tracking_outputs = []
        self.response_outputs = []
        self.response_resolver: ResponseResolver | None = None
        self.running = False
        self.latest_frame: TrackingFrame | None = None
        self.latest_response: LocalizationResponse | None = None

    def configure_input(self, adapter) -> None:
        self.input_adapter = adapter

    def configure_response_resolver(self, resolver: ResponseResolver | None) -> None:
        self.response_resolver = resolver

    def add_tracking_output(self, adapter) -> None:
        self.tracking_outputs.append(adapter)

    def add_response_output(self, adapter) -> None:
        self.response_outputs.append(adapter)

    # V0.7 compatibility: add_output means raw tracking output.
    def add_output(self, adapter) -> None:
        self.add_tracking_output(adapter)

    def clear_outputs(self) -> None:
        self.tracking_outputs.clear()
        self.response_outputs.clear()

    def start(self) -> None:
        if self.input_adapter is None:
            raise RuntimeError("TrackingEngine requires an input adapter.")

        started = []
        try:
            for output in [*self.tracking_outputs, *self.response_outputs]:
                output.start()
                started.append(output)
        except Exception:
            for output in reversed(started):
                output.stop()
            raise

        self.running = True

    def process_results(
        self,
        results: Iterable,
    ) -> tuple[list[TrackingFrame], list[LocalizationResponse]]:
        """Normalize, resolve, and publish the current adapter results."""
        if self.input_adapter is None:
            raise RuntimeError("TrackingEngine has no configured input adapter.")

        results = list(results)

        if results:
            frames = [
                self.input_adapter.to_tracking_frame(result)
                for result in results
            ]
        else:
            frames = [self.input_adapter.invalid_tracking_frame()]

        responses: list[LocalizationResponse] = []
        for frame in frames:
            response = (
                self.response_resolver.resolve(frame)
                if self.response_resolver is not None
                else LocalizationResponse.invalid(
                    source=frame.source,
                    target_id=frame.target_id,
                )
            )
            responses.append(response)

            if self.running:
                for output in self.tracking_outputs:
                    output.send(frame)
                for output in self.response_outputs:
                    output.send(response)

        self.latest_frame = frames[0] if frames else None
        self.latest_response = responses[0] if responses else None
        return frames, responses

    def calibrate_current(self) -> float:
        if self.response_resolver is None:
            raise RuntimeError("No response resolver is configured.")
        if self.latest_frame is None:
            raise RuntimeError("No tracking sample is available yet.")
        return self.response_resolver.calibrate(self.latest_frame)

    def stop(self) -> None:
        self.running = False
        for output in [*self.tracking_outputs, *self.response_outputs]:
            try:
                output.stop()
            except Exception:
                pass

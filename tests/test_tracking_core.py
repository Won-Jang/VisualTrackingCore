"""Dependency-light tests for tracking and localization response core."""

import unittest

from core import ResponseResolver, TrackingEngine, TrackingFrame, wrap_degrees


class DummyResult:
    marker_id = 7


class DummyInput:
    def to_tracking_frame(self, result):
        return TrackingFrame(
            source="Dummy",
            target_id=result.marker_id,
            tracking_valid=True,
            confidence=0.9,
            x=12.0,
            y=-3.0,
            position_unit="px",
            yaw_deg=25.0,
        )

    def invalid_tracking_frame(self):
        return TrackingFrame.invalid("Dummy", target_id=7, position_unit="px")


class DummyOutput:
    def __init__(self):
        self.started = False
        self.samples = []

    def start(self):
        self.started = True

    def send(self, sample):
        self.samples.append(sample)

    def stop(self):
        self.started = False


class TrackingCoreTests(unittest.TestCase):
    def make_engine(self):
        engine = TrackingEngine()
        engine.configure_input(DummyInput())
        engine.configure_response_resolver(
            ResponseResolver(orientation_axis="yaw")
        )
        return engine

    def test_valid_result_is_normalized_and_published(self):
        engine = self.make_engine()
        output = DummyOutput()
        engine.add_tracking_output(output)
        engine.start()

        frames, responses = engine.process_results([DummyResult()])

        self.assertEqual(len(frames), 1)
        self.assertTrue(frames[0].tracking_valid)
        self.assertEqual(frames[0].target_id, 7)
        self.assertEqual(output.samples[0].x, 12.0)
        self.assertFalse(responses[0].response_valid)

    def test_no_detection_emits_invalid_tracking_and_response(self):
        engine = self.make_engine()
        frames, responses = engine.process_results([])

        self.assertEqual(len(frames), 1)
        self.assertFalse(frames[0].tracking_valid)
        self.assertEqual(frames[0].position_unit, "px")
        self.assertFalse(responses[0].response_valid)

    def test_calibration_converts_yaw_to_relative_azimuth(self):
        resolver = ResponseResolver(orientation_axis="yaw")
        reference = TrackingFrame(
            source="Dummy", tracking_valid=True, confidence=1.0, yaw_deg=10.0
        )
        resolver.calibrate(reference)

        frame = TrackingFrame(
            source="Dummy", tracking_valid=True, confidence=0.8, yaw_deg=42.5
        )
        response = resolver.resolve(frame)

        self.assertTrue(response.response_valid)
        self.assertAlmostEqual(response.response_azimuth_deg, 32.5)
        self.assertAlmostEqual(response.confidence, 0.8)

    def test_angle_wrap_crosses_180_cleanly(self):
        self.assertAlmostEqual(wrap_degrees(190.0), -170.0)
        self.assertAlmostEqual(wrap_degrees(-190.0), 170.0)

    def test_marker_style_roll_can_drive_response(self):
        resolver = ResponseResolver(orientation_axis="roll")
        resolver.calibrate(
            TrackingFrame(source="Marker", tracking_valid=True, roll_deg=-5.0)
        )
        response = resolver.resolve(
            TrackingFrame(source="Marker", tracking_valid=True, roll_deg=15.0)
        )
        self.assertAlmostEqual(response.response_azimuth_deg, 20.0)


if __name__ == "__main__":
    unittest.main()

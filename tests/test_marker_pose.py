"""Synthetic geometry and detector integration; no physical camera required."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from adapters.apriltag_adapter import AprilTagAdapter
from adapters.aruco_adapter import ArUcoAdapter
from core.camera_calibration import CameraCalibration
from core.planar_marker_pose import PlanarMarkerPoseEstimator
from core import ResponseResolver, TrackingEngine, TrackingFrame
from main import VisualTrackingApp
from outputs import LSLOutputAdapter, LSLResponseOutputAdapter
from tools.calibrate_camera import make_board, write_board, matched_points, calibrate_observations


MATRIX = np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1.]])
DISTORTION = np.array([0.03, -0.01, 0.001, -0.002, 0.0])


def calibration():
    return CameraCalibration(MATRIX, DISTORTION, 640, 480, 0.2)


def rotation(yaw=0., pitch=0., roll=0.):
    # Independent construction of the documented overhead rotation sequence.
    z, x, y = np.deg2rad([yaw, pitch, roll])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return rz @ rx @ ry @ np.diag([1., -1., -1.])


def projected(estimator, yaw=0., pitch=0., roll=0., translation=(15., -20., 600.)):
    rvec, _ = cv2.Rodrigues(rotation(yaw, pitch, roll))
    corners, _ = cv2.projectPoints(estimator.object_points, rvec, np.array(translation),
                                   estimator.camera_matrix, estimator.dist_coeffs)
    return corners.reshape(4, 2)


class PoseTests(unittest.TestCase):
    def setUp(self):
        self.estimator = PlanarMarkerPoseEstimator(MATRIX, DISTORTION, 50)

    def test_synthetic_heading_with_pitch_and_roll(self):
        for yaw in (-179., -90., -30., 0., 30., 90., 179.):
            for pitch, roll in ((0., 0.), (20., -15.), (-25., 30.)):
                with self.subTest(yaw=yaw, pitch=pitch, roll=roll):
                    pose = self.estimator.estimate(projected(self.estimator, yaw, pitch, roll))
                    self.assertIsNotNone(pose)
                    np.testing.assert_allclose([pose.x, pose.y, pose.z], [15, -20, 600], atol=1e-3)
                    np.testing.assert_allclose([pose.yaw_deg, pose.pitch_deg, pose.roll_deg],
                                               [yaw, pitch, roll], atol=1e-3)
                    self.assertEqual(pose.position_unit, "mm")
                    self.assertLess(pose.reprojection_error_px, 1e-4)

    def test_size_controls_translation_scale(self):
        points = projected(self.estimator, 30, 15, 20)
        doubled = PlanarMarkerPoseEstimator(MATRIX, DISTORTION, 100).estimate(points)
        self.assertAlmostEqual(doubled.z, 1200., places=3)
        self.assertAlmostEqual(doubled.yaw_deg, 30., places=3)

    def test_invalid_geometry(self):
        for corners in (None, [], [[1, 2]] * 4, np.zeros((3, 2)),
                        np.full((4, 2), np.nan), [[0, 0], [10, 10], [0, 10], [10, 0]],
                        [[0, 0], [1, 0], [2, 0], [3, 0]]):
            with self.subTest(corners=corners):
                self.assertIsNone(self.estimator.estimate(corners))

    def test_invalid_size_and_solver_failure(self):
        for size in (0, -1, np.inf, np.nan):
            with self.assertRaises(ValueError):
                PlanarMarkerPoseEstimator(MATRIX, DISTORTION, size)
        with patch("core.planar_marker_pose.cv2.solvePnPGeneric", side_effect=cv2.error("failed")):
            self.assertIsNone(self.estimator.estimate(projected(self.estimator)))

    def test_excessive_reprojection_error_rejected(self):
        points = projected(self.estimator, 20, 10, 15)
        points[0] += [30, -20]
        self.assertIsNone(self.estimator.estimate(points))


class CalibrationTests(unittest.TestCase):
    def test_roundtrip_and_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.npz"
            calibration().save(path)
            loaded = CameraCalibration.load(path)
            loaded.validate_resolution(640, 480)
            np.testing.assert_equal(loaded.camera_matrix, MATRIX)
            with self.assertRaisesRegex(ValueError, "does not match"):
                loaded.validate_resolution(1280, 720)

    def test_missing_corrupt_and_unsafe_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.npz"
            with self.assertRaises(ValueError):
                CameraCalibration.load(path)
            np.savez(path, camera_matrix=MATRIX)
            with self.assertRaisesRegex(ValueError, "Missing calibration fields"):
                CameraCalibration.load(path)
            path.write_text("not an npz")
            with self.assertRaises(ValueError):
                CameraCalibration.load(path)
            np.savez(path, camera_matrix=MATRIX, dist_coeffs=DISTORTION,
                     image_width=640, image_height=480, reprojection_error=0.1,
                     board_definition=np.array({"unsafe": True}, dtype=object))
            with self.assertRaises(ValueError):
                CameraCalibration.load(path)

    def test_invalid_numeric_fields(self):
        cases = [(np.eye(2), DISTORTION, 640, 480, .2),
                 (MATRIX * np.nan, DISTORTION, 640, 480, .2),
                 (MATRIX, [0, 0, 0], 640, 480, .2),
                 (MATRIX, np.zeros((2, 2)), 640, 480, .2),
                 (MATRIX, DISTORTION, 0, 480, .2),
                 (MATRIX, DISTORTION, 640.5, 480, .2),
                 (MATRIX, DISTORTION, 640, 480, np.nan)]
        for case in cases:
            with self.assertRaises(ValueError):
                CameraCalibration(*case)

    def test_generated_board_detects_and_has_physical_size(self):
        from PIL import Image
        board = make_board()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "board.png"
            write_board(board, path)
            with Image.open(path) as image:
                self.assertAlmostEqual(image.width / image.info["dpi"][0] * 25.4, 170, delta=.1)
                self.assertAlmostEqual(image.height / image.info["dpi"][1] * 25.4, 230, delta=.1)
            corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(cv2.imread(str(path)))
            self.assertEqual(len(ids), 24)
            self.assertIsNotNone(matched_points(board, corners, ids))
            self.assertIsNone(matched_points(board, corners[:3], ids[:3]))

    def test_synthetic_calibration_recovers_intrinsics(self):
        board = make_board()
        objects = board.getChessboardCorners()
        all_objects, all_images = [], []
        rng = np.random.default_rng(4)
        for _ in range(20):
            rvec = rng.uniform(-.35, .35, 3)
            tvec = np.array([rng.uniform(-120, -30), rng.uniform(-130, -20), rng.uniform(500, 850)])
            image, _ = cv2.projectPoints(objects, rvec, tvec, MATRIX, DISTORTION)
            all_objects.append(objects)
            all_images.append(image.reshape(-1, 2))
        result = calibrate_observations(all_objects, all_images, (640, 480), {"test": True})
        np.testing.assert_allclose(result.camera_matrix, MATRIX, atol=.05)
        self.assertLess(result.reprojection_error, .001)
        with self.assertRaises(ValueError):
            calibrate_observations(all_objects[:3], all_images[:3], (640, 480), {})


class AdapterTests(unittest.TestCase):
    def test_both_detectors_normalize_corners_into_same_pose(self):
        estimator = PlanarMarkerPoseEstimator(MATRIX, DISTORTION, 50)
        points = projected(estimator, 35, 20, -10)
        frame = np.zeros((480, 640, 3), np.uint8)
        aruco = ArUcoAdapter(tracking_mode="3d_pose", calibration=calibration())
        aruco.detector = Mock()
        aruco.detector.detectMarkers.return_value = ([points.reshape(1, 4, 2)], np.array([[0]]), [])
        with patch("adapters.apriltag_adapter.Detector"):
            april = AprilTagAdapter(tracking_mode="3d_pose", calibration=calibration())
        april.detector.detect.return_value = [Mock(tag_id=0, center=points.mean(axis=0),
                                                  corners=points[[3, 2, 1, 0]])]
        for adapter in (april, aruco):
            result = adapter.detect(frame)[0]
            normalized = adapter.to_tracking_frame(result)
            self.assertTrue(normalized.tracking_valid)
            self.assertEqual(normalized.position_unit, "mm")
            self.assertAlmostEqual(normalized.yaw_deg, 35, places=3)
            self.assertAlmostEqual(normalized.z, 600, places=3)
            with self.assertRaisesRegex(ValueError, "does not match"):
                adapter.detect(np.zeros((240, 320, 3), np.uint8))
            adapter.pose_estimator.estimate = Mock(return_value=None)
            failed = adapter.to_tracking_frame(adapter.detect(frame)[0])
            self.assertFalse(failed.tracking_valid)
            self.assertIsNone(failed.roll_deg)
            self.assertIsNone(failed.x)

    def test_actual_aruco_detection_and_2d_compatibility(self):
        frame = np.full((480, 640, 3), 255, np.uint8)
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        pixels = cv2.aruco.generateImageMarker(dictionary, 0, 160)
        frame[160:320, 240:400] = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        adapter = ArUcoAdapter()
        result = adapter.detect(frame)[0]
        normalized = adapter.to_tracking_frame(result)
        self.assertEqual(normalized.position_unit, "px")
        self.assertIsNone(normalized.yaw_deg)
        self.assertAlmostEqual(normalized.roll_deg, 0)
        adapter3d = ArUcoAdapter(tracking_mode="3d_pose", calibration=calibration())
        result3d = adapter3d.detect(frame)[0]
        self.assertTrue(result3d.pose_valid)
        self.assertAlmostEqual(result3d.yaw_deg, 0, delta=1)

    def test_actual_apriltag_corner_order(self):
        path = Path(__file__).resolve().parents[1] / "markers" / "tag41_12_00000.png"
        marker = cv2.imread(str(path))
        marker = cv2.resize(marker, (260, 260), interpolation=cv2.INTER_NEAREST)
        frame = np.full((480, 640, 3), 255, np.uint8)
        frame[110:370, 190:450] = marker
        adapter = AprilTagAdapter(tracking_mode="3d_pose", calibration=calibration())
        results = adapter.detect(frame)
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].pose_valid)
        self.assertAlmostEqual(results[0].yaw_deg, 0, delta=1)
        # A clockwise image rotation should increase overhead yaw.
        rotated = cv2.warpAffine(frame, cv2.getRotationMatrix2D((320, 240), -35, 1),
                                 (640, 480), borderValue=(255, 255, 255))
        self.assertAlmostEqual(adapter.detect(rotated)[0].yaw_deg, 35, delta=1)

    def test_loss_after_calibration_and_required_file(self):
        with self.assertRaisesRegex(ValueError, "requires"):
            ArUcoAdapter(tracking_mode="3d_pose")
        adapter = ArUcoAdapter(tracking_mode="3d_pose", calibration=calibration())
        engine = TrackingEngine()
        engine.configure_input(adapter)
        resolver = ResponseResolver()
        resolver.calibrate(TrackingFrame(source="ArUco", target_id=0, tracking_valid=True, yaw_deg=5))
        engine.configure_response_resolver(resolver)
        frames, responses = engine.process_results(adapter.detect(np.zeros((480, 640, 3), np.uint8)))
        self.assertFalse(frames[0].tracking_valid)
        self.assertFalse(responses[0].response_valid)
        self.assertEqual(frames[0].position_unit, "mm")


class ResponseAndOutputTests(unittest.TestCase):
    def test_mode_selection_and_zero_reference(self):
        for solution, mode, axis in (("ArUco", "2d_rotation", "roll"),
                                     ("AprilTag", "3d_pose", "yaw"),
                                     ("ArUco", "3d_pose", "yaw"),
                                     ("MediaPipe Face", "2d_rotation", "yaw")):
            resolver = VisualTrackingApp._response_resolver_for_solution(solution, mode)
            self.assertEqual(resolver.orientation_axis, axis)
            ref = TrackingFrame(source=solution, target_id=0, tracking_valid=True)
            setattr(ref, axis + "_deg", 170)
            resolver.calibrate(ref)
            setattr(ref, axis + "_deg", -170)
            self.assertEqual(resolver.resolve(ref).response_azimuth_deg, 20)
            setattr(ref, axis + "_deg", np.nan)
            self.assertFalse(resolver.resolve(ref).response_valid)
            with self.assertRaises(ValueError):
                resolver.calibrate(ref)
            setattr(ref, axis + "_deg", 10)
            ref.target_id = 1
            self.assertFalse(resolver.resolve(ref).response_valid)

    def test_sign_wrap_stays_half_open(self):
        resolver = ResponseResolver(sign=-1)
        resolver.calibrate(TrackingFrame(tracking_valid=True, yaw_deg=0))
        response = resolver.resolve(TrackingFrame(tracking_valid=True, yaw_deg=180))
        self.assertEqual(response.response_azimuth_deg, -180)

    def test_lsl_schema_metadata_and_invalid_samples(self):
        import pylsl
        for output, channel_count in ((LSLOutputAdapter(position_unit="mm", metadata={"tracking_mode": "3d_pose"}), 9),
                                      (LSLResponseOutputAdapter(metadata={"tracking_mode": "3d_pose"}), 5)):
            with patch.object(pylsl, "StreamOutlet") as outlet:
                output.start()
                info = outlet.call_args.args[0]
                self.assertEqual(info.channel_count(), channel_count)
                self.assertIn("3d_pose", info.as_xml())
                if channel_count == 9:
                    self.assertIn("<unit>mm</unit>", info.as_xml())
                    output.send(TrackingFrame.invalid("ArUco", 0, "mm"))
                else:
                    from core.localization_response import LocalizationResponse
                    output.send(LocalizationResponse.invalid("ArUco", target_id=0))
                sample = outlet.return_value.push_sample.call_args.args[0]
                self.assertTrue(np.isnan(sample[0]))
                self.assertEqual(sample[-2], 0.)
                self.assertEqual(sample[-1], 0.)
                output.stop()


if __name__ == "__main__":
    unittest.main()

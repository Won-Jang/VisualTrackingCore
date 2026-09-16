"""Exercise real Tk controls with a fake camera; no window or webcam is opened."""

from pathlib import Path
import tempfile
import tkinter as tk
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from core.camera_calibration import CameraCalibration
from main import VisualTrackingApp


class AppWorkflowTests(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Tk display unavailable: {exc}")
        self.root.withdraw()
        self.addCleanup(self.root.destroy)
        self.app = VisualTrackingApp(self.root)
        self.addCleanup(self.app.stop_tracking)
        self.app.camera = Mock()
        self.app.lsl_enabled_var.set(False)
        self.errors = patch("main.messagebox.showerror").start()
        self.addCleanup(patch.stopall)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "test.npz"
        CameraCalibration(np.array([[800., 0, 320], [0, 800., 240], [0, 0, 1.]]),
                          np.zeros(5), 640, 480, .1).save(self.path)
        self.app.solution_var.set("ArUco")
        self.app.mode_var.set("3D Pose")
        self.app.camera_calibration_path_var.set(str(self.path))
        self.app._on_solution_changed()

    def test_3d_start_calibrate_stop_restart(self):
        marker = cv2.aruco.generateImageMarker(
            cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 0, 160)
        frame = np.full((480, 640, 3), 255, np.uint8)
        frame[160:320, 240:400] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
        self.app.camera.read.return_value = (True, frame)
        self.app.start_tracking()
        self.errors.assert_not_called()
        self.assertTrue(self.app.running)
        self.assertEqual(str(self.app.mode_combo.cget("state")), "disabled")
        self.assertEqual(self.app.engine.latest_frame.position_unit, "mm")
        self.assertIsNotNone(self.app._after_id)
        self.app.calibrate_reference()
        self.assertTrue(self.app.engine.response_resolver.calibrated)
        first_timer = self.app._after_id
        self.app.stop_tracking()
        self.assertIsNone(self.app._after_id)
        self.assertNotIn(first_timer, self.root.tk.call("after", "info"))
        self.app.start_tracking()
        self.assertFalse(self.app.engine.response_resolver.calibrated)
        self.errors.assert_not_called()

    def test_mismatched_resolution_blocks_before_lsl_start(self):
        self.app.lsl_enabled_var.set(True)
        self.app.camera.read.return_value = (True, np.zeros((240, 320, 3), np.uint8))
        with patch("main.LSLOutputAdapter") as output:
            self.app.start_tracking()
            output.assert_not_called()
        self.assertFalse(self.app.running)
        self.assertIsNone(self.app.adapter)
        self.app.camera.close.assert_called()
        self.assertIn("does not match", self.errors.call_args.args[1])

    def test_invalid_file_blocks_camera_and_2d_needs_no_file(self):
        self.app.camera_calibration_path_var.set(str(self.path) + ".missing")
        self.app.start_tracking()
        self.app.camera.open.assert_not_called()
        self.assertFalse(self.app.running)
        self.app.mode_var.set("2D Marker Rotation")
        self.app._on_solution_changed()
        self.app.camera.read.return_value = (True, np.zeros((480, 640, 3), np.uint8))
        self.errors.reset_mock()
        self.app.start_tracking()
        self.assertTrue(self.app.running)
        self.assertEqual(self.app.engine.latest_frame.position_unit, "px")
        self.errors.assert_not_called()

    def test_mediapipe_ignores_marker_controls(self):
        self.app.solution_var.set("MediaPipe Face")
        self.app._on_solution_changed()
        self.assertEqual(str(self.app.mode_combo.cget("state")), "disabled")
        self.assertFalse(self.app.pose_controls.winfo_manager())
        self.assertEqual(self.app.response_source_var.get(), "Response source: Yaw")


if __name__ == "__main__":
    unittest.main()

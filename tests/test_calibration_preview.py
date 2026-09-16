"""Regression checks for OpenCV 5 corner layout in the actual capture preview."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from tools.calibrate_camera import (
    capture_calibration, make_board, normalize_charuco_detection,
)


class CalibrationPreviewTests(unittest.TestCase):
    def test_both_opencv_layouts_can_be_drawn(self):
        for shape, id_shape in (((3, 2), (3,)), ((3, 1, 2), (3, 1))):
            corners = np.array([[10, 10], [20, 20], [30, 30]], np.float32).reshape(shape)
            ids = np.arange(3, dtype=np.int32).reshape(id_shape)
            points, identifiers = normalize_charuco_detection(corners, ids)
            self.assertEqual(points.shape, (3, 1, 2))
            self.assertEqual(identifiers.shape, (3, 1))
            frame = np.zeros((60, 60, 3), np.uint8)
            cv2.aruco.drawDetectedCornersCharuco(frame, points, identifiers)
            self.assertTrue(frame.any())

    def test_missing_empty_and_mismatched_detection_is_skipped(self):
        for corners, ids in ((None, None), (np.empty((0, 2)), np.empty(0)),
                             (np.zeros((3, 2)), np.arange(2)),
                             (np.full((3, 2), np.nan), np.arange(3))):
            self.assertEqual(normalize_charuco_detection(corners, ids), (None, None))

    def test_real_detection_and_drawing_through_capture_loop(self):
        board = make_board()
        pixels = board.generateImage((500, 700), marginSize=20)
        frame = cv2.cvtColor(pixels, cv2.COLOR_GRAY2BGR)
        camera = Mock()
        camera.isOpened.return_value = True
        camera.read.side_effect = [(True, np.zeros_like(frame)), (True, frame.copy()),
                                   (True, frame.copy())]
        args = SimpleNamespace(camera=0, width=None, height=None)
        # Keep real detectBoard, matchImagePoints, and corner drawing. Only
        # replace camera/window I/O so the exact failure path runs headlessly.
        with patch("tools.calibrate_camera.cv2.VideoCapture", return_value=camera), \
                patch("tools.calibrate_camera.cv2.imshow") as show, \
                patch("tools.calibrate_camera.cv2.waitKey", side_effect=[32, 32, ord("q")]), \
                patch("tools.calibrate_camera.cv2.getWindowProperty", return_value=1), \
                patch("tools.calibrate_camera.cv2.destroyAllWindows"), \
                patch("builtins.print"):
            capture_calibration(args, board)
            self.assertEqual(show.call_count, 3)
        camera.release.assert_called_once()


if __name__ == "__main__":
    unittest.main()

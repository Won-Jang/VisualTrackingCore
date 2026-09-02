# OpenCV ArUco marker-tracking adapter.

import cv2
import math
import numpy as np

from adapters.base import TrackingAdapter
from tracking_types import TrackingResult


class ArUcoAdapter(TrackingAdapter):
    # Detect OpenCV ArUco markers and return normalized tracking results.

    name = "ArUco"

    def __init__(
        self,
        target_id: int | None = 0,
        dictionary_id: int = cv2.aruco.DICT_4X4_50,
    ):
        super().__init__(target_id=target_id)

        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    def detect(self, frame) -> list[TrackingResult]:
        """Detect configured ArUco markers in one BGR camera frame."""
        corners, ids, _rejected = self.detector.detectMarkers(frame)

        if ids is None:
            return []

        height, width = frame.shape[:2]
        frame_center_x = width / 2.0
        frame_center_y = height / 2.0

        results = []

        for marker_corners, marker_id in zip(corners, ids.flatten()):
            marker_id = int(marker_id)

            if self.target_id is not None and marker_id != self.target_id:
                continue

            pts = marker_corners.reshape((4, 2))

            center_x = float(np.mean(pts[:, 0]))
            center_y = float(np.mean(pts[:, 1]))

            top_left = pts[0]
            top_right = pts[1]

            dx = float(top_right[0] - top_left[0])
            dy = float(top_right[1] - top_left[1])
            angle = math.degrees(math.atan2(dy, dx))

            relative_x = center_x - frame_center_x
            relative_y = frame_center_y - center_y

            results.append(
                TrackingResult(
                    marker_id=marker_id,
                    x_px=relative_x,
                    y_px=relative_y,
                    angle_deg=angle,
                    center_px=(center_x, center_y),
                    corners_px=[
                        (float(point[0]), float(point[1]))
                        for point in pts
                    ],
                )
            )

        return results

# AprilTag marker-tracking adapter

import cv2
import math
from pupil_apriltags import Detector

from adapters.base import TrackingAdapter
from tracking_types import TrackingResult


class AprilTagAdapter(TrackingAdapter):
    # Detect AprilTags and convert detections to the shared result format.

    name = "AprilTag"

    def __init__(
        self,
        target_id: int | None = 0,
        family: str = "tagStandard41h12",
    ):
        super().__init__(target_id=target_id)

        self.detector = Detector(
            families=family,
            nthreads=2,
            quad_decimate=1.0,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

    def detect(self, frame) -> list[TrackingResult]:
        # Detect configured AprilTags in one BGR camera frame.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray)

        height, width = frame.shape[:2]
        frame_center_x = width / 2.0
        frame_center_y = height / 2.0

        results = []

        for tag in detections:
            marker_id = int(tag.tag_id)

            if self.target_id is not None and marker_id != self.target_id:
                continue

            center_x = float(tag.center[0])
            center_y = float(tag.center[1])

            corners = [
                (float(point[0]), float(point[1]))
                for point in tag.corners
            ]

            # Corner 0 -> corner 1 defines the tag's top-edge direction.
            dx = corners[1][0] - corners[0][0]
            dy = corners[1][1] - corners[0][1]
            angle = math.degrees(math.atan2(dy, dx))

            # Common coordinate system:
            # 0,0 = camera image center
            # +X = right
            # +Y = up
            relative_x = center_x - frame_center_x
            relative_y = frame_center_y - center_y

            results.append(
                TrackingResult(
                    marker_id=marker_id,
                    x_px=relative_x,
                    y_px=relative_y,
                    angle_deg=angle,
                    center_px=(center_x, center_y),
                    corners_px=corners,
                )
            )

        return results

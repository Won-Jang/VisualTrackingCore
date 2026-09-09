# OpenCV ArUco marker-tracking adapter
#
# This module implements the ArUco input adapter used by VisualTrackingCore.
# Its responsibility is intentionally limited to the tracking layer:
#
#   1. Detect ArUco markers in each camera frame.
#   2. Optionally filter detections to a specific marker ID.
#   3. Measure the marker center position in the camera image.
#   4. Measure the marker's 2-D image-plane rotation.
#   5. Convert those values into the shared TrackingResult format.
#
# The adapter does NOT perform speaker-array calibration or calculate the final
# localization response angle. Those operations belong to the higher-level
# VisualTrackingCore response/calibration layer so that AprilTag, ArUco,
# MediaPipe, PointTracker, and future adapters can share the same workflow.

import cv2
import math
import numpy as np

from adapters.base import TrackingAdapter
from tracking_types import TrackingResult


class ArUcoAdapter(TrackingAdapter):
    """
    Detect OpenCV ArUco markers and return normalized tracking results.

    Coordinate convention
    ---------------------
    OpenCV image coordinates normally use:

        origin = upper-left corner
        +X     = right
        +Y     = down

    VisualTrackingCore converts the detected marker center to:

        origin = camera image center
        +X     = right
        +Y     = up

    The angle returned by this adapter is the marker's 2-D in-plane rotation
    within the camera image. It is not a full 3-D head-pose estimate.

    For the planned overhead-camera configuration, this image-plane rotation
    can serve as the orientation signal that is later calibrated against the
    speaker-array 0-degree reference.
    """

    # Name shown in the adapter-selection UI.
    name = "ArUco"

    def __init__(
        self,
        target_id: int | None = 0,
        dictionary_id: int = cv2.aruco.DICT_4X4_50,
    ):
        """
        Initialize the ArUco detector.

        Parameters
        ----------
        target_id:
            Marker ID that VisualTrackingCore should track.

            Example:
                target_id=0
                    Track only ArUco marker ID 0.

                target_id=None
                    Accept every ArUco marker detected in the frame.

        dictionary_id:
            OpenCV ArUco dictionary used by the printed marker.

            The default dictionary is DICT_4X4_50, which contains
            50 unique markers using a 4x4 internal binary pattern.

            The dictionary selected here must match the dictionary used
            when generating/printing the physical ArUco marker.
        """

        # Initialize common adapter settings, including target_id.
        super().__init__(target_id=target_id)

        # Load the predefined ArUco dictionary.
        #
        # Both the detector and the printed marker must use the same
        # dictionary or the marker will not be decoded correctly.
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)

        # Create the standard OpenCV ArUco detector settings.
        #
        # We currently use OpenCV defaults. These parameters can later be
        # tuned if we need better performance for:
        #   - small markers
        #   - greater camera distances
        #   - low lighting
        #   - motion blur
        #   - highly angled tags
        parameters = cv2.aruco.DetectorParameters()

        # Create the detector once during initialization.
        #
        # Reusing the same detector for every frame is more efficient than
        # recreating it inside detect().
        self.detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters,
        )

    def detect(self, frame) -> list[TrackingResult]:
        """
        Detect configured ArUco markers in one OpenCV camera frame.

        Parameters
        ----------
        frame:
            One camera frame supplied by OpenCV.

        Returns
        -------
        list[TrackingResult]
            A list containing the accepted marker detections.

            If no matching marker is detected, an empty list is returned.

            If target_id is None, more than one TrackingResult may be returned
            when multiple ArUco markers are visible.
        """

        # Ask OpenCV to detect and decode ArUco markers in the current frame.
        #
        # corners:
        #     Corner coordinates for each successfully decoded marker.
        #
        # ids:
        #     Marker IDs corresponding to the entries in corners.
        #
        # _rejected:
        #     Candidate regions that looked like markers but could not be
        #     decoded. We do not currently use these, but they can be useful
        #     later when debugging detection quality.
        corners, ids, _rejected = self.detector.detectMarkers(frame)

        # OpenCV returns ids=None when no valid ArUco markers are detected.
        #
        # Returning an empty list is the shared adapter convention for
        # "nothing detected in this frame." Higher VisualTrackingCore layers
        # handle tracking-loss state from there.
        if ids is None:
            return []

        # Determine the image dimensions and center point.
        #
        # The image center becomes the origin of the shared tracking
        # coordinate system.
        height, width = frame.shape[:2]
        frame_center_x = width / 2.0
        frame_center_y = height / 2.0

        # Each accepted marker detection will be converted into one
        # TrackingResult and appended here.
        results = []

        # OpenCV returns one corner array per marker and one marker ID per
        # corner array. Flatten ids to simplify iteration.
        for marker_corners, marker_id in zip(corners, ids.flatten()):

            # Convert NumPy integer types to a normal Python int.
            marker_id = int(marker_id)

            # If the adapter is configured to track one specific marker,
            # ignore every other detected ID.
            #
            # If target_id is None, this filter is disabled and all detected
            # markers are accepted.
            if self.target_id is not None and marker_id != self.target_id:
                continue

            # OpenCV typically returns ArUco corners with shape (1, 4, 2).
            #
            # Reshape to a simpler 4x2 array:
            #
            #     [[x0, y0],
            #      [x1, y1],
            #      [x2, y2],
            #      [x3, y3]]
            #
            # OpenCV orders ArUco corners clockwise beginning at the marker's
            # top-left corner:
            #
            #     0 = top-left
            #     1 = top-right
            #     2 = bottom-right
            #     3 = bottom-left
            pts = marker_corners.reshape((4, 2))

            # Calculate the marker center by averaging all four corner
            # positions.
            #
            # This is generally more stable than using only one edge or corner.
            center_x = float(np.mean(pts[:, 0]))
            center_y = float(np.mean(pts[:, 1]))

            # Use the marker's top edge to determine its 2-D image rotation.
            #
            # Vector:
            #     top-left -> top-right
            #
            # When this edge is perfectly horizontal, angle is approximately
            # 0 degrees. Rotating the physical marker changes this angle.
            top_left = pts[0]
            top_right = pts[1]

            dx = float(top_right[0] - top_left[0])
            dy = float(top_right[1] - top_left[1])

            # atan2() calculates the direction of the top-edge vector.
            #
            # math.degrees() converts the result from radians to degrees.
            #
            # IMPORTANT:
            # This is IMAGE-PLANE marker rotation. It should not be interpreted
            # as a full 3-D roll measurement.
            angle = math.degrees(math.atan2(dy, dx))

            # Convert the marker center from normal OpenCV image coordinates
            # into the shared VisualTrackingCore coordinate system.
            #
            # OpenCV:
            #     origin = upper-left
            #     +Y     = down
            #
            # VisualTrackingCore:
            #     origin = image center
            #     +X     = right
            #     +Y     = up
            #
            # X only needs to be shifted relative to the image center.
            relative_x = center_x - frame_center_x

            # Y is both shifted and inverted because image coordinates increase
            # downward while our shared coordinate system increases upward.
            relative_y = frame_center_y - center_y

            # Convert this detector-specific result into the common
            # TrackingResult structure used by VisualTrackingCore.
            #
            # This keeps the rest of the application independent of whether
            # the tracking source is ArUco, AprilTag, MediaPipe, or another
            # future adapter.
            results.append(
                TrackingResult(
                    # ID encoded by the physical ArUco marker.
                    marker_id=marker_id,

                    # Marker center relative to the camera-image center.
                    # Units are pixels.
                    x_px=relative_x,
                    y_px=relative_y,

                    # Marker's 2-D image-plane rotation in degrees.
                    angle_deg=angle,

                    # Original absolute image-space center.
                    # Retained for UI drawing and possible future processing.
                    center_px=(center_x, center_y),

                    # Original corner positions in image coordinates.
                    # Retained for drawing the marker boundary and future
                    # calculations such as pose estimation.
                    corners_px=[
                        (float(point[0]), float(point[1]))
                        for point in pts
                    ],
                )
            )

        # If no marker passed the target-ID filter, results will be empty.
        #
        # Tracking-loss handling is intentionally left to the higher-level
        # VisualTrackingCore engine rather than being implemented separately
        # inside each adapter.
        return results

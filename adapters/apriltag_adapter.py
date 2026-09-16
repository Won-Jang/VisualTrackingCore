# AprilTag marker-tracking adapter
#
# This module provides the AprilTag implementation used by VisualTrackingCore.
# Its responsibility is intentionally limited to:
#   1. Detecting AprilTags in a camera frame.
#   2. Selecting the requested tag ID.
#   3. Converting the detected tag position/orientation into TrackingResult.
#
# The adapter does NOT perform experiment-level calibration or calculate the
# final localization response angle. Those operations belong to the core
# response/calibration layer so that all tracking methods can share the same
# experiment-facing behavior.

import cv2
import math
from pupil_apriltags import Detector

from adapters.marker_base import MarkerTrackingAdapter
from tracking_types import TrackingResult


class AprilTagAdapter(MarkerTrackingAdapter):
    """
    Detect AprilTags and convert each valid detection into TrackingResult.

    The returned coordinates use the shared VisualTrackingCore image-space
    convention:

        image center = (0, 0)
        +X           = right
        +Y           = up

    The returned angle is the tag's 2-D in-plane rotation in the camera image.
    In the overhead-camera use case this value can later be used by the
    response resolver as the participant's horizontal orientation signal.
    """

    # Name displayed in the VisualTrackingCore adapter-selection UI.
    name = "AprilTag"

    def __init__(
        self,
        target_id: int | None = 0,
        family: str = "tagStandard41h12",
        tracking_mode="2d_rotation",
        calibration=None,
        marker_size_mm=50.0,
    ):
        """
        Initialize the AprilTag detector.

        Parameters
        ----------
        target_id:
            AprilTag ID to track. If None, all detected tags are returned.
            The default is tag ID 0.

        family:
            AprilTag family used by the printed marker.
            VisualTrackingCore currently defaults to tagStandard41h12.
        """

        # Store common adapter settings such as target_id in the base class.
        super().__init__(target_id, tracking_mode, calibration, marker_size_mm)

        # Create the detector once when the adapter is initialized.
        # Reusing it for every frame is more efficient than rebuilding it
        # inside detect().
        self.detector = Detector(
            families=family,

            # Number of CPU threads available to the detector.
            nthreads=2,

            # 1.0 keeps the original image resolution during quad detection.
            # Higher values can improve speed but may make small/far tags
            # harder to detect.
            quad_decimate=1.0,

            # Gaussian blur applied before quad detection.
            # 0.0 means no additional blur.
            quad_sigma=0.0,

            # Refine detected edges for more accurate corner locations.
            refine_edges=1,

            # Sharpening used during tag decoding.
            decode_sharpening=0.25,

            # Disable detector debug-image output.
            debug=0,
        )

    def detect(self, frame) -> list[TrackingResult]:
        """
        Detect AprilTags in one OpenCV BGR camera frame.

        Parameters
        ----------
        frame:
            One camera frame in OpenCV BGR format.

        Returns
        -------
        list[TrackingResult]
            Zero or more normalized tracking results. If target_id is set,
            only the matching tag is returned.
        """

        # AprilTag detection operates on grayscale images, while OpenCV
        # webcam frames are normally supplied in BGR format.
        self.validate_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Run the detector for the current frame.
        detections = self.detector.detect(gray)

        # Find the image center. TrackingResult reports marker position
        # relative to this point instead of absolute image coordinates.
        height, width = frame.shape[:2]
        frame_center_x = width / 2.0
        frame_center_y = height / 2.0

        # Each accepted detection will become one TrackingResult.
        results = []

        for tag in detections:
            # Convert detector output to a normal Python integer.
            marker_id = int(tag.tag_id)

            # If a target tag is configured, ignore all other IDs.
            # Setting target_id=None intentionally allows all tags.
            if self.target_id is not None and marker_id != self.target_id:
                continue

            # AprilTag center coordinates use normal image coordinates:
            #   origin = upper-left
            #   +X     = right
            #   +Y     = down
            center_x = float(tag.center[0])
            center_y = float(tag.center[1])

            # Convert corner coordinates into plain Python tuples.
            # These values are retained for drawing overlays and for possible
            # future pose-estimation work.
            corners = [
                (float(point[0]), float(point[1]))
                for point in tag.corners
            ]

            # Estimate the tag's 2-D rotation within the camera image.
            #
            # Corner 0 -> corner 1 defines one marker edge. atan2() gives the
            # direction of that edge relative to the image X axis.
            #
            # IMPORTANT:
            # This is IMAGE-PLANE marker rotation, not a full 3-D roll angle.
            # In the overhead-camera configuration, this rotation can serve as
            # the orientation signal that is later calibrated to the
            # speaker-array 0-degree reference.
            dx = corners[1][0] - corners[0][0]
            dy = corners[1][1] - corners[0][1]
            angle = math.degrees(math.atan2(dy, dx))

            # Convert normal image coordinates to the shared
            # VisualTrackingCore coordinate convention:
            #
            #   (0, 0) = camera image center
            #   +X     = right
            #   +Y     = up
            #
            # Since image-space Y increases downward, the Y calculation is
            # intentionally inverted here.
            relative_x = center_x - frame_center_x
            relative_y = frame_center_y - center_y

            # Package AprilTag-specific measurements into the common
            # TrackingResult structure so higher layers can process all
            # adapters through the same interface.
            results.append(
                TrackingResult(
                    marker_id=marker_id,

                    # Marker center relative to the camera-image center.
                    x_px=relative_x,
                    y_px=relative_y,

                    # 2-D marker rotation in degrees.
                    angle_deg=angle,

                    # Original image coordinates retained for UI overlays and
                    # future calculations.
                    center_px=(center_x, center_y),
                    corners_px=corners,
                )
            )

        # Returning an empty list means the configured target was not detected
        # in this frame. Tracking-loss handling belongs to higher core layers.
        # Pupil/AprilTag native order is BL, BR, TR, TL in decoded tag space.
        # Normalize to TL, TR, BR, BL only for the shared pose estimator.
        return self.apply_pose(results, (3, 2, 1, 0))

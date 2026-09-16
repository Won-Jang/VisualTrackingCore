"""Mode and pose handling shared by the two planar-marker detectors."""

from adapters.base import TrackingAdapter
from core.planar_marker_pose import PlanarMarkerPoseEstimator
from core.tracking_frame import TrackingFrame


class MarkerTrackingAdapter(TrackingAdapter):
    def __init__(self, target_id=0, tracking_mode="2d_rotation",
                 calibration=None, marker_size_mm=50.0):
        super().__init__(target_id)
        if tracking_mode not in ("2d_rotation", "3d_pose"):
            raise ValueError("Tracking mode must be 2d_rotation or 3d_pose.")
        self.tracking_mode = tracking_mode
        self.calibration = calibration
        self.pose_estimator = None
        self.position_unit = "px"
        if tracking_mode == "3d_pose":
            if calibration is None:
                raise ValueError("3D Pose requires a valid camera calibration file.")
            self.pose_estimator = PlanarMarkerPoseEstimator(
                calibration.camera_matrix, calibration.dist_coeffs, marker_size_mm
            )
            self.position_unit = "mm"

    def validate_frame(self, frame):
        if self.pose_estimator is not None:
            height, width = frame.shape[:2]
            self.calibration.validate_resolution(width, height)

    def apply_pose(self, results, corner_order):
        if self.pose_estimator is None:
            return results
        for result in results:
            # Preserve detector-native corners/2D angle for existing overlays.
            corners = [result.corners_px[i] for i in corner_order]
            pose = self.pose_estimator.estimate(corners)
            result.pose_valid = pose is not None
            if pose is not None:
                result.source_x, result.source_y, result.source_z = pose.x, pose.y, pose.z
                result.yaw_deg, result.pitch_deg, result.roll_deg = (
                    pose.yaw_deg, pose.pitch_deg, pose.roll_deg
                )
                result.pose_origin_px, result.pose_axes_px = pose.origin_px, pose.axes_px
                result.pose_reprojection_error_px = pose.reprojection_error_px
        return results

    def to_tracking_frame(self, result):
        if self.tracking_mode == "3d_pose" and result.pose_valid is not True:
            return TrackingFrame.invalid(self.name, result.marker_id, "mm")
        return super().to_tracking_frame(result)

    def invalid_tracking_frame(self):
        return TrackingFrame.invalid(self.name, self.target_id, self.position_unit)

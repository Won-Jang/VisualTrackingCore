"""Shared square-marker pose estimation for an overhead head-tracking camera.

Camera translation: X right, Y down, Z away from the camera, in millimeters.
Marker corners: decoded top-left, top-right, bottom-right, bottom-left (never
sort by image position). Marker X points right, Y toward its printed top,
Z out of the printed face. Mount printed top toward the participant's front.

An upright, flat, face-on marker has R0 = diag(1, -1, -1). Decompose
R @ R0.T = Rz(yaw) @ Rx(pitch) @ Ry(roll). Overhead yaw is clockwise in the
image, about camera Z, not the camera-Y yaw used by a front-facing face model.
This extracts heading from the 3D marker forward vector, not a slanted edge.
Camera must point vertically down; an oblique camera needs extrinsic calibration.
"""

from dataclasses import dataclass
import math

import cv2
import numpy as np

from core.camera_calibration import validate_intrinsics


ORIENTATION_CONVENTION = "overhead_Rz_yaw_Rx_pitch_Ry_roll"


@dataclass
class MarkerPose:
    x: float
    y: float
    z: float
    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    rotation_matrix: np.ndarray
    reprojection_error_px: float
    origin_px: tuple
    axes_px: list
    position_unit: str = "mm"


def overhead_angles(rotation_matrix):
    rotation = rotation_matrix @ np.diag([1.0, -1.0, -1.0])
    # Heading is undefined when the marker's forward axis points into the camera.
    if math.hypot(rotation[0, 1], rotation[1, 1]) < 1e-6:
        return None
    yaw = math.atan2(-rotation[0, 1], rotation[1, 1])
    pitch = math.asin(float(np.clip(rotation[2, 1], -1, 1)))
    roll = math.atan2(-rotation[2, 0], rotation[2, 2])
    return tuple(math.degrees(angle) for angle in (yaw, pitch, roll))


class PlanarMarkerPoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs, marker_size_mm,
                 max_reprojection_error_px=3.0):
        self.camera_matrix, self.dist_coeffs = validate_intrinsics(camera_matrix, dist_coeffs)
        self.marker_size_mm = float(marker_size_mm)
        if not math.isfinite(self.marker_size_mm) or self.marker_size_mm <= 0:
            raise ValueError("Marker size must be finite and greater than zero (mm).")
        self.max_reprojection_error_px = float(max_reprojection_error_px)
        if (not math.isfinite(self.max_reprojection_error_px)
                or self.max_reprojection_error_px <= 0):
            raise ValueError("Reprojection-error limit must be positive and finite.")
        half = self.marker_size_mm / 2
        # OpenCV IPPE_SQUARE requires this exact object-point order.
        self.object_points = np.array([
            [-half, half, 0], [half, half, 0],
            [half, -half, 0], [-half, -half, 0],
        ], dtype=np.float64)

    def estimate(self, corners):
        """Return a checked pose, or None for missing/degenerate/failed estimates."""
        try:
            points = np.asarray(corners, dtype=np.float64)
            if points.shape not in ((4, 2), (1, 4, 2), (4, 1, 2)):
                return None
            points = np.ascontiguousarray(points.reshape(4, 2))
            contour = points.astype(np.float32)
            if (not np.isfinite(points).all() or not cv2.isContourConvex(contour)
                    or abs(cv2.contourArea(contour)) < 4):
                return None
            # Planar pose has two solutions. Score physically valid candidates by
            # reprojection error. A single marker cannot eliminate all ambiguity.
            candidates = []
            result = cv2.solvePnPGeneric(self.object_points, points, self.camera_matrix,
                                        self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if result[0]:
                candidates.extend(zip(result[1], result[2]))
            # Iterative homography initialization also handles exactly frontal
            # configurations where IPPE may return nonfinite candidates.
            ok, rvec, tvec = cv2.solvePnP(self.object_points, points, self.camera_matrix,
                                        self.dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                candidates.append((rvec, tvec))
            valid = []
            for rvec, tvec in candidates:
                if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
                    continue
                rotation, _ = cv2.Rodrigues(rvec)
                camera_points = (rotation @ self.object_points.T).T + tvec.reshape(3)
                if np.any(camera_points[:, 2] <= 0):
                    continue
                # Printed front must face the camera, not away from it.
                if np.dot(rotation[:, 2], tvec.reshape(3)) >= 0:
                    continue
                projected, _ = cv2.projectPoints(self.object_points, rvec, tvec,
                                                self.camera_matrix, self.dist_coeffs)
                error = float(np.sqrt(np.mean(np.sum(
                    (projected.reshape(4, 2) - points) ** 2, axis=1))))
                angles = overhead_angles(rotation)
                if (angles is not None and math.isfinite(error)
                        and error <= self.max_reprojection_error_px):
                    valid.append((error, rvec, tvec, rotation, angles))
            if not valid:
                return None
            error, rvec, tvec, rotation, angles = min(valid, key=lambda item: item[0])
            length = self.marker_size_mm / 2
            axes = np.array([[0, 0, 0], [length, 0, 0], [0, length, 0], [0, 0, length]])
            projected, _ = cv2.projectPoints(axes, rvec, tvec,
                                            self.camera_matrix, self.dist_coeffs)
            projected = projected.reshape(-1, 2)
            if not np.isfinite(projected).all():
                return None
            return MarkerPose(*map(float, tvec.reshape(3)), *angles, rotation, error,
                              tuple(projected[0]), [tuple(p) for p in projected[1:]])
        except (ValueError, TypeError, cv2.error):
            return None

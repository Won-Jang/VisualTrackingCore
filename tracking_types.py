# Common data structures exchanged between tracking adapters and the UI.

from dataclasses import dataclass


@dataclass
class TrackingResult:
    # Common tracking result returned by every adapter.

    marker_id: int
    x_px: float
    y_px: float
    angle_deg: float
    center_px: tuple[float, float]
    corners_px: list[tuple[float, float]]

    # Optional 3D head-pose fields. Marker-based adapters can leave these None.
    yaw_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None

    # Optional raw source translation values (for external 6-DOF adapters).
    # No unit conversion is applied by VisualTrackingCore.
    source_x: float | None = None
    source_y: float | None = None
    source_z: float | None = None

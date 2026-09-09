# Shared OpenCV drawing helpers for visualizing tracking results.

import cv2


def draw_tracking_overlay(frame, result, trail=None, orientation_label=None):
    # Draw center, bounds, orientation, pose axes, values, and motion trail.

    height, width = frame.shape[:2]
    frame_center = (width // 2, height // 2)

    # Camera image center.
    cv2.drawMarker(
        frame,
        frame_center,
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=30,
        thickness=2,
    )

    if result is not None:
        corners = [
            (int(x), int(y))
            for x, y in result.corners_px
        ]

        for i in range(len(corners)):
            cv2.line(
                frame,
                corners[i],
                corners[(i + 1) % len(corners)],
                (0, 255, 0),
                2,
            )

        center = (
            int(result.center_px[0]),
            int(result.center_px[1]),
        )

        cv2.circle(frame, center, 6, (0, 0, 255), -1)

        # Existing 2D orientation line retained for marker adapters and
        # backward compatibility.
        if len(corners) >= 2:
            edge_mid = (
                int((corners[0][0] + corners[1][0]) / 2),
                int((corners[0][1] + corners[1][1]) / 2),
            )
            cv2.line(frame, center, edge_mid, (255, 0, 0), 3)

        # MediaPipe may provide projected 3D pose axes.
        pose_origin = getattr(result, "pose_origin_px", None)
        pose_axes = getattr(result, "pose_axes_px", None)

        if pose_origin is not None and pose_axes:
            origin = (int(pose_origin[0]), int(pose_origin[1]))
            axis_points = [
                (int(point[0]), int(point[1]))
                for point in pose_axes
            ]

            # X, Y, Z axes.
            cv2.line(frame, origin, axis_points[0], (0, 0, 255), 3)
            cv2.line(frame, origin, axis_points[1], (0, 255, 0), 3)
            cv2.line(frame, origin, axis_points[2], (255, 0, 0), 3)

        lines = [
            f"ID: {result.marker_id}",
            f"X: {result.x_px:+.0f}px   Y: {result.y_px:+.0f}px",
        ]

        if orientation_label is not None:
            lines.append(
                f"{orientation_label}: {result.angle_deg:+.1f} deg"
            )

        if (
            result.yaw_deg is not None
            and result.pitch_deg is not None
            and result.roll_deg is not None
        ):
            lines.append(
                f"Yaw: {result.yaw_deg:+.1f}   "
                f"Pitch: {result.pitch_deg:+.1f}   "
                f"Roll: {result.roll_deg:+.1f}"
            )

        for i, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (20, 40 + i * 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

    if trail:
        for i in range(1, len(trail)):
            cv2.line(
                frame,
                trail[i - 1],
                trail[i],
                (255, 255, 0),
                2,
            )

    return frame

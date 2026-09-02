# MediaPipe Face Landmarker adapter for markerless head tracking.

import math
import time
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from adapters.base import TrackingAdapter
from tracking_types import TrackingResult


class MediaPipeFaceAdapter(TrackingAdapter):
    # Markerless face/head tracking using MediaPipe Face Landmarker.

    name = "MediaPipe Face"

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    )

    # Generic 3D face model points in millimeters.
    #
    # The corresponding MediaPipe landmarks are:
    #   1   nose tip
    #   152 chin
    #   33  left eye outer corner
    #   263 right eye outer corner
    #   61  left mouth corner
    #   291 right mouth corner
    #
    # These points are intentionally a generic POC model rather than
    # subject-specific anatomical calibration.
    MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),          # nose tip
            (0.0, -63.6, -12.5),      # chin
            (-43.3, 32.7, -26.0),     # left eye outer
            (43.3, 32.7, -26.0),      # right eye outer
            (-28.9, -28.9, -24.1),    # left mouth corner
            (28.9, -28.9, -24.1),     # right mouth corner
        ],
        dtype=np.float64,
    )

    LANDMARK_IDS = (1, 152, 33, 263, 61, 291)

    def __init__(
        self,
        target_id: int | None = 0,
        model_path: str | None = None,
    ):
        # Preserve the common factory signature. MediaPipe tracks one face
        # and reports it internally as ID 0.
        super().__init__(target_id=0)

        if model_path is None:
            model_path = (
                Path(__file__).resolve().parent.parent
                / "models"
                / "face_landmarker.task"
            )

        self.model_path = Path(model_path)
        self._ensure_model()

        base_options = mp.tasks.BaseOptions(
            model_asset_path=str(self.model_path)
        )
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )
        self._last_timestamp_ms = 0

    def _ensure_model(self) -> None:
        # Download the Face Landmarker model on first use when necessary.
        if self.model_path.exists():
            return

        self.model_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            urllib.request.urlretrieve(
                self.MODEL_URL,
                str(self.model_path),
            )
        except Exception as exc:
            try:
                self.model_path.unlink(missing_ok=True)
            except Exception:
                pass

            raise RuntimeError(
                "MediaPipe Face Landmarker model is missing and could not "
                "be downloaded automatically.\n\n"
                f"Expected file:\n{self.model_path}\n\n"
                "Connect this computer to the internet once and start the "
                "MediaPipe adapter again, or manually place "
                "'face_landmarker.task' in the models folder."
            ) from exc

    def _timestamp_ms(self) -> int:
        """Return a strictly increasing millisecond timestamp for VIDEO mode."""
        timestamp = int(time.monotonic() * 1000)

        # MediaPipe VIDEO mode requires strictly increasing timestamps.
        if timestamp <= self._last_timestamp_ms:
            timestamp = self._last_timestamp_ms + 1

        self._last_timestamp_ms = timestamp
        return timestamp

    @staticmethod
    def _rotation_matrix_to_euler(rotation_matrix):
        # Return pitch, yaw, roll in degrees from a 3x3 rotation matrix.

        sy = math.sqrt(
            rotation_matrix[0, 0] ** 2
            + rotation_matrix[1, 0] ** 2
        )

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(
                rotation_matrix[2, 1],
                rotation_matrix[2, 2],
            )
            y = math.atan2(
                -rotation_matrix[2, 0],
                sy,
            )
            z = math.atan2(
                rotation_matrix[1, 0],
                rotation_matrix[0, 0],
            )
        else:
            x = math.atan2(
                -rotation_matrix[1, 2],
                rotation_matrix[1, 1],
            )
            y = math.atan2(
                -rotation_matrix[2, 0],
                sy,
            )
            z = 0.0

        return (
            math.degrees(x),
            math.degrees(y),
            math.degrees(z),
        )

    def _estimate_head_pose(self, points, width, height):
        # Estimate approximate 3D head orientation using OpenCV solvePnP.
        image_points = np.array(
            [points[index] for index in self.LANDMARK_IDS],
            dtype=np.float64,
        )

        # Approximate camera intrinsics from image dimensions.
        # This is suitable for POC verification; later camera calibration
        # can replace this matrix for more accurate angular measurements.
        focal_length = float(width)
        camera_matrix = np.array(
            [
                [focal_length, 0.0, width / 2.0],
                [0.0, focal_length, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.zeros((4, 1), dtype=np.float64)

        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.MODEL_POINTS,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            return None

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pitch, yaw, roll = self._rotation_matrix_to_euler(rotation_matrix)

        # Pose axes projected into the camera image. These are attached to the
        # result dynamically so the generic overlay can draw them without
        # changing marker-based adapters.
        axis_length = 45.0
        axis_3d = np.array(
            [
                (axis_length, 0.0, 0.0),
                (0.0, axis_length, 0.0),
                (0.0, 0.0, axis_length),
            ],
            dtype=np.float64,
        )

        projected, _ = cv2.projectPoints(
            axis_3d,
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion,
        )

        origin, _ = cv2.projectPoints(
            np.array([(0.0, 0.0, 0.0)], dtype=np.float64),
            rotation_vector,
            translation_vector,
            camera_matrix,
            distortion,
        )

        origin = tuple(origin.reshape(2))
        axes = [tuple(point) for point in projected.reshape(-1, 2)]

        return pitch, yaw, roll, origin, axes

    def detect(self, frame) -> list[TrackingResult]:
        # Track one face and return its center, bounding box, and head pose.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        detection = self.landmarker.detect_for_video(
            mp_image,
            self._timestamp_ms(),
        )

        if not detection.face_landmarks:
            return []

        landmarks = detection.face_landmarks[0]

        height, width = frame.shape[:2]
        frame_center_x = width / 2.0
        frame_center_y = height / 2.0

        points = [
            (
                float(landmark.x * width),
                float(landmark.y * height),
            )
            for landmark in landmarks
        ]

        xs = [point[0] for point in points]
        ys = [point[1] for point in points]

        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)

        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        # Preserve the original 2D eye-line roll as angle_deg for backward
        # compatibility with the existing UI/output contract.
        left_eye = points[33]
        right_eye = points[263]

        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = math.degrees(math.atan2(dy, dx))

        relative_x = center_x - frame_center_x
        relative_y = frame_center_y - center_y

        corners = [
            (min_x, min_y),
            (max_x, min_y),
            (max_x, max_y),
            (min_x, max_y),
        ]

        pose = self._estimate_head_pose(points, width, height)

        pitch = yaw = roll = None
        pose_origin = None
        pose_axes = None

        if pose is not None:
            pitch, yaw, roll, pose_origin, pose_axes = pose

        result = TrackingResult(
            marker_id=0,
            x_px=relative_x,
            y_px=relative_y,
            angle_deg=angle,
            center_px=(center_x, center_y),
            corners_px=corners,
            yaw_deg=yaw,
            pitch_deg=pitch,
            roll_deg=roll,
        )

        # Optional visualization metadata; common consumers can ignore it.
        result.pose_origin_px = pose_origin
        result.pose_axes_px = pose_axes

        return [result]

    def close(self) -> None:
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
            self.landmarker = None

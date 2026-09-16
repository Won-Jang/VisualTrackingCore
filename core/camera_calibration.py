"""Validated pinhole-camera calibration files; never load pickled objects."""

from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np


def validate_intrinsics(camera_matrix, dist_coeffs):
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    distortion = np.asarray(dist_coeffs, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera_matrix must be a finite 3 x 3 matrix.")
    if (matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or not np.allclose([matrix[0, 1], matrix[1, 0]], 0)):
        raise ValueError("camera_matrix must have positive focal lengths and standard pinhole form.")
    if (distortion.ndim not in (1, 2)
            or (distortion.ndim == 2 and 1 not in distortion.shape)
            or distortion.size not in (4, 5, 8, 12, 14)
            or not np.isfinite(distortion).all()):
        raise ValueError("dist_coeffs must be a finite vector of 4, 5, 8, 12, or 14 coefficients.")
    return matrix.copy(), distortion.reshape(-1, 1).copy()


@dataclass
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_width: int
    image_height: int
    reprojection_error: float
    timestamp: str = ""
    board_definition: str = ""

    def __post_init__(self):
        self.camera_matrix, self.dist_coeffs = validate_intrinsics(
            self.camera_matrix, self.dist_coeffs
        )
        for field in ("image_width", "image_height"):
            value = float(getattr(self, field))
            if not np.isfinite(value) or value <= 0 or not value.is_integer():
                raise ValueError(f"{field} must be a positive integer.")
            setattr(self, field, int(value))
        self.reprojection_error = float(self.reprojection_error)
        if not np.isfinite(self.reprojection_error) or self.reprojection_error < 0:
            raise ValueError("reprojection_error must be finite and nonnegative.")

    def validate_resolution(self, width, height):
        if (width, height) != (self.image_width, self.image_height):
            raise ValueError(
                f"Calibration resolution {self.image_width} x {self.image_height} "
                f"does not match camera frame {width} x {height}. "
                "Use the calibrated resolution or recalibrate; no automatic scaling is applied."
            )

    @classmethod
    def load(cls, path):
        required = ("camera_matrix", "dist_coeffs", "image_width", "image_height",
                    "reprojection_error")
        try:
            with np.load(path, allow_pickle=False) as data:
                missing = [key for key in required if key not in data]
                if missing:
                    raise ValueError("Missing calibration fields: " + ", ".join(missing))
                values = {key: data[key] for key in required}
                for key in ("image_width", "image_height", "reprojection_error"):
                    values[key] = values[key].item()
                for key in ("timestamp", "board_definition"):
                    values[key] = str(data[key].item()) if key in data else ""
                return cls(**values)
        except (OSError, ValueError, TypeError, AttributeError, EOFError, BadZipFile) as exc:
            raise ValueError(f"Cannot load camera calibration '{path}': {exc}") from exc

    def save(self, path):
        path = Path(path)
        if path.suffix.lower() != ".npz":
            raise ValueError("Calibration output must end in .npz.")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, camera_matrix=self.camera_matrix, dist_coeffs=self.dist_coeffs,
                 image_width=self.image_width, image_height=self.image_height,
                 reprojection_error=self.reprojection_error, timestamp=self.timestamp,
                 board_definition=self.board_definition)

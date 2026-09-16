"""Generate a ChArUco board and capture a pinhole-camera calibration.

Run from the project root: python tools/calibrate_camera.py --help
No camera is opened on import or when generating the printable board.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.camera_calibration import CameraCalibration


def make_board(squares_x=5, squares_y=7, square_mm=30.0, marker_mm=22.0):
    if squares_x < 3 or squares_y < 3 or squares_x * squares_y // 2 > 50:
        raise ValueError("Board needs at least 3 x 3 squares and at most 50 markers.")
    if not (np.isfinite(square_mm) and np.isfinite(marker_mm)
            and 0 < marker_mm < square_mm):
        raise ValueError("Lengths must be finite, with 0 < marker mm < square mm.")
    return cv2.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm, marker_mm,
        cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
    )


def write_board(board, path, dpi=300):
    """Write a PNG with physical resolution metadata and a 10 mm white margin."""
    path = Path(path)
    if path.suffix.lower() != ".png":
        raise ValueError("Board output must end in .png.")
    squares_x, squares_y = board.getChessboardSize()
    square_mm = board.getSquareLength()
    # Render the board without margins to avoid a non-square aspect fit.
    size = tuple(round(n * square_mm / 25.4 * dpi) for n in (squares_x, squares_y))
    pixels = board.generateImage(size, marginSize=0, borderBits=1)
    margin = round(10 / 25.4 * dpi)
    canvas = Image.new("L", (size[0] + 2 * margin, size[1] + 2 * margin), 255)
    canvas.paste(Image.fromarray(pixels), (margin, margin))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, dpi=(dpi, dpi))


def normalize_charuco_detection(corners, ids):
    """Normalize OpenCV 4/5 detector results for the drawing/calibration APIs.

    OpenCV 5 can return N x 2 corners and N IDs. Its drawing API still
    requires two-channel points (N x 1 x 2), not an N x 2 scalar matrix.
    """
    if corners is None or ids is None:
        return None, None
    points = np.asarray(corners, dtype=np.float32)
    identifiers = np.asarray(ids, dtype=np.int32)
    if (points.ndim not in (2, 3) or points.shape[-1] != 2
            or (points.ndim == 3 and points.shape[1] != 1)
            or points.size == 0 or identifiers.size != points.shape[0]
            or not np.isfinite(points).all()):
        return None, None
    return (np.ascontiguousarray(points.reshape(-1, 1, 2)),
            np.ascontiguousarray(identifiers.reshape(-1, 1)))


def matched_points(board, corners, ids):
    """Reject insufficient or collinear captures before calibration."""
    if corners is None or ids is None or len(ids) < 6:
        return None
    objects, images = board.matchImagePoints(corners, ids)
    objects = np.asarray(objects, dtype=np.float32).reshape(-1, 3)
    images = np.asarray(images, dtype=np.float32).reshape(-1, 2)
    if np.linalg.matrix_rank(objects[:, :2] - objects[:, :2].mean(axis=0)) < 2:
        return None
    return objects, images


def calibrate_observations(object_points, image_points, image_size, board_definition):
    if len(object_points) < 15 or len(object_points) != len(image_points):
        raise ValueError("Capture at least 15 valid, varied board views before saving.")
    error, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return CameraCalibration(matrix, distortion, *image_size, error,
                             datetime.now(timezone.utc).isoformat(),
                             json.dumps(board_definition, sort_keys=True))


def capture_calibration(args, board):
    camera = cv2.VideoCapture(args.camera)
    objects, images = [], []
    image_size = None
    detector = cv2.aruco.CharucoDetector(board)
    status = "SPACE: capture | C: calibrate/save (15+ views) | Q/ESC: cancel"
    window = "VisualTrackingCore - Camera calibration"
    try:
        if not camera.isOpened():
            raise RuntimeError(f"Cannot open camera {args.camera}.")
        if args.width is not None:
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        while True:
            ok, frame = camera.read()
            if not ok:
                raise RuntimeError("Cannot read camera frame.")
            size = (frame.shape[1], frame.shape[0])
            if image_size is not None and size != image_size:
                raise RuntimeError("Camera resolution changed during calibration; restart capture.")
            image_size = size
            corners, ids, _, _ = detector.detectBoard(frame)
            corners, ids = normalize_charuco_detection(corners, ids)
            observation = matched_points(board, corners, ids)
            if ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(frame, corners, ids)
            cv2.putText(frame, f"{size[0]}x{size[1]} | Captured: {len(objects)} | "
                        f"Corners: {0 if ids is None else len(ids)}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.putText(frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (0, 255, 0), 1)
            cv2.imshow(window, frame)
            key = cv2.waitKey(10) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                print("Calibration cancelled; no file saved.")
                return
            if key == ord(" "):
                if observation is None:
                    status = "Need 6+ non-collinear ChArUco corners. Move board and retry."
                elif any(np.array_equal(observation[0], old_obj)
                         and np.sqrt(np.mean((observation[1] - old_img) ** 2)) < 5
                         for old_obj, old_img in zip(objects, images)):
                    status = "View too similar. Change board position, tilt, or distance."
                else:
                    objects.append(observation[0])
                    images.append(observation[1])
                    status = "Captured. Move/tilt board. SPACE: capture | C: save | Q: cancel"
            if key == ord("c"):
                if len(objects) < 15:
                    status = "Need 15+ varied views; aim for 20-25 across the image."
                    continue
                definition = dict(squares_x=args.squares_x, squares_y=args.squares_y,
                                  square_mm=args.square_mm, marker_mm=args.marker_mm,
                                  dictionary="DICT_4X4_50", opencv_version=cv2.__version__,
                                  valid_frames=len(objects))
                calibration = calibrate_observations(objects, images, image_size, definition)
                calibration.save(args.output)
                print(f"Resolution: {image_size[0]} x {image_size[1]}\n"
                      f"Valid frames: {len(objects)}\n"
                      f"Reprojection RMS: {calibration.reprojection_error:.4f} px\n"
                      f"Saved: {Path(args.output).resolve()}")
                return
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generate-board", metavar="BOARD.png", help="Generate board and exit")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--squares-x", type=int, default=5)
    parser.add_argument("--squares-y", type=int, default=7)
    parser.add_argument("--square-mm", type=float, default=30.0)
    parser.add_argument("--marker-mm", type=float, default=22.0)
    parser.add_argument("--output", default="calibration/webcam_calibration.npz")
    args = parser.parse_args()
    try:
        if ((args.width is None) != (args.height is None)
                or (args.width is not None and (args.width <= 0 or args.height <= 0))):
            raise ValueError("Specify both --width and --height as positive integers.")
        board = make_board(args.squares_x, args.squares_y, args.square_mm, args.marker_mm)
        if args.generate_board:
            write_board(board, args.generate_board)
            print(f"Saved: {args.generate_board}\nPrint at actual size/100%; "
                  f"measure each square: {args.square_mm:g} mm. Do not fit to page.")
        else:
            if Path(args.output).suffix.lower() != ".npz":
                raise ValueError("Calibration output must end in .npz.")
            capture_calibration(args, board)
    except (ValueError, RuntimeError, OSError, cv2.error) as exc:
        parser.exit(1, f"Calibration error: {exc}\n")


if __name__ == "__main__":
    main()

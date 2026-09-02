# Small OpenCV camera wrapper used by camera-based tracking adapters.

import cv2


class Camera:
    # Own one OpenCV VideoCapture and provide safe open/read/close methods.

    def __init__(self):
        self.capture = None
        self.index = None

    @property
    def is_open(self) -> bool:
        return self.capture is not None and self.capture.isOpened()

    def open(self, index: int) -> None:
        # Open a camera index, closing any previously opened device first.
        self.close()

        self.capture = cv2.VideoCapture(index)
        self.index = index

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"Could not open camera {index}")

    def read(self):
        # Return the next OpenCV frame, or ``(False, None)`` if closed.
        if not self.is_open:
            return False, None

        return self.capture.read()

    def close(self) -> None:
        # Release the current camera device and reset wrapper state.
        if self.capture is not None:
            self.capture.release()

        self.capture = None
        self.index = None

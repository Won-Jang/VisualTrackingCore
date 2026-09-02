# Tkinter entry point and main UI for the VisualTrackingCore proof of concept.

import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageTk

from adapters import adapter_names, create_adapter
from camera import Camera
from overlay import draw_tracking_overlay


class VisualTrackingApp:
    # Main desktop UI that selects, runs, and visualizes tracking adapters.

    def __init__(self, root):
        self.root = root
        self.root.title("VisualTrackingCore")
        self.root.geometry("1050x760")

        self.camera = Camera()
        self.adapter = None
        self.running = False
        self.trail = deque(maxlen=50)

        self.solution_var = tk.StringVar(value=adapter_names()[0])
        self.camera_var = tk.StringVar(value="0")
        self.udp_port_var = tk.StringVar(value="4242")
        self.target_id_var = tk.StringVar(value="0")

        self.status_var = tk.StringVar(value="Stopped")
        self.position_var = tk.StringVar(value="No detection")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self):
        # Create the controls, status line, and video/visualization area.
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill="x")

        ttk.Label(controls, text="Tracking solution:").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )

        self.solution_combo = ttk.Combobox(
            controls,
            textvariable=self.solution_var,
            values=adapter_names(),
            state="readonly",
            width=18,
        )
        self.solution_combo.grid(row=0, column=1, padx=5, pady=5)
        self.solution_combo.bind(
            "<<ComboboxSelected>>",
            self._on_solution_changed,
        )

        self.input_label = ttk.Label(controls, text="Camera index:")
        self.input_label.grid(
            row=0, column=2, padx=5, pady=5, sticky="w"
        )

        self.camera_entry = ttk.Entry(
            controls,
            textvariable=self.camera_var,
            width=6,
        )
        self.camera_entry.grid(row=0, column=3, padx=5, pady=5)

        self.udp_port_entry = ttk.Entry(
            controls,
            textvariable=self.udp_port_var,
            width=6,
        )
        self.udp_port_entry.grid(row=0, column=3, padx=5, pady=5)
        self.udp_port_entry.grid_remove()

        self.target_id_label = ttk.Label(controls, text="Target ID:")
        self.target_id_label.grid(
            row=0, column=4, padx=5, pady=5, sticky="w"
        )

        self.target_id_entry = ttk.Entry(
            controls,
            textvariable=self.target_id_var,
            width=6,
        )
        self.target_id_entry.grid(row=0, column=5, padx=5, pady=5)

        self.start_button = ttk.Button(
            controls,
            text="Start",
            command=self.start_tracking,
        )
        self.start_button.grid(row=0, column=6, padx=10, pady=5)

        self.stop_button = ttk.Button(
            controls,
            text="Stop",
            command=self.stop_tracking,
            state="disabled",
        )
        self.stop_button.grid(row=0, column=7, padx=5, pady=5)

        ttk.Button(
            controls,
            text="Clear Trail",
            command=self.trail.clear,
        ).grid(row=0, column=8, padx=5, pady=5)

        info = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        info.pack(fill="x")

        ttk.Label(info, textvariable=self.status_var).pack(side="left")
        ttk.Label(info, text="   |   ").pack(side="left")
        ttk.Label(info, textvariable=self.position_var).pack(side="left")

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        self._on_solution_changed()

    def _on_solution_changed(self, _event=None):
        # Show only the controls relevant to the selected adapter.
        solution = self.solution_var.get()
        is_mediapipe = solution == "MediaPipe Face"
        is_opentrack = solution == "OpenTrack UDP"

        if is_opentrack:
            self.input_label.config(text="UDP port:")
            self.camera_entry.grid_remove()
            self.udp_port_entry.grid()
        else:
            self.input_label.config(text="Camera index:")
            self.udp_port_entry.grid_remove()
            self.camera_entry.grid()

        if is_mediapipe or is_opentrack:
            self.target_id_entry.config(state="disabled")
            self.target_id_label.config(text="Target ID: (N/A)")
        else:
            self.target_id_entry.config(state="normal")
            self.target_id_label.config(text="Target ID:")

    def _parse_target_id(self):
        # Parse a numeric marker ID; blank or ``all`` means no filtering.
        text = self.target_id_var.get().strip()

        if text == "" or text.lower() == "all":
            return None

        return int(text)

    def start_tracking(self):
        # Create the selected adapter and start the UI update loop.
        if self.running:
            return

        solution = self.solution_var.get()
        is_opentrack = solution == "OpenTrack UDP"

        try:
            if solution in ("MediaPipe Face", "OpenTrack UDP"):
                target_id = 0
            else:
                target_id = self._parse_target_id()

            if is_opentrack:
                # OpenTrack owns its webcam, so this app only opens a UDP listener.
                udp_port = int(self.udp_port_var.get())
                if not 1 <= udp_port <= 65535:
                    raise ValueError("UDP port must be between 1 and 65535.")

                self.adapter = create_adapter(
                    solution,
                    target_id=target_id,
                    port=udp_port,
                )
                input_text = f"UDP 127.0.0.1:{udp_port}"
            else:
                # Camera-based adapters process frames captured by this application.
                camera_index = int(self.camera_var.get())
                self.adapter = create_adapter(
                    solution,
                    target_id=target_id,
                )
                self.camera.open(camera_index)
                input_text = f"Camera {camera_index}"

        except Exception as exc:
            if self.adapter is not None:
                self.adapter.close()
                self.adapter = None

            self.camera.close()
            messagebox.showerror("Start failed", str(exc))
            return

        self.running = True
        self.trail.clear()

        self.status_var.set(f"Running: {solution} / {input_text}")

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.solution_combo.config(state="disabled")

        self._update_frame()

    def stop_tracking(self):
        # Stop tracking and release camera/adapter resources.
        self.running = False

        self.camera.close()

        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None

        self.status_var.set("Stopped")
        self.position_var.set("No detection")

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.solution_combo.config(state="readonly")

    @staticmethod
    def _opentrack_visualization_frame():
        # Create a synthetic canvas because OpenTrack displays its own webcam.
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        height, width = frame.shape[:2]

        # Simple coordinate grid for external OpenTrack data.
        for x in range(0, width, 80):
            cv2.line(frame, (x, 0), (x, height), (45, 45, 45), 1)
        for y in range(0, height, 60):
            cv2.line(frame, (0, y), (width, y), (45, 45, 45), 1)

        cv2.putText(
            frame,
            "OpenTrack UDP - external camera/tracker",
            (20, height - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (180, 180, 180),
            2,
        )
        cv2.putText(
            frame,
            "Configure OpenTrack Output: UDP over network -> 127.0.0.1:4242",
            (20, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
        )
        return frame

    def _update_frame(self):
        # Acquire/detect/draw one frame and schedule the next UI refresh.
        if not self.running:
            return

        is_opentrack = self.solution_var.get() == "OpenTrack UDP"

        if is_opentrack:
            frame = self._opentrack_visualization_frame()
        else:
            ok, frame = self.camera.read()

            if not ok:
                self.stop_tracking()
                messagebox.showerror("Camera error", "Could not read camera frame.")
                return

        try:
            results = self.adapter.detect(frame)
        except Exception as exc:
            self.stop_tracking()
            messagebox.showerror("Tracking error", str(exc))
            return

        # The current POC UI visualizes one primary result at a time.
        result = results[0] if results else None

        if result is not None:
            center = (
                int(result.center_px[0]),
                int(result.center_px[1]),
            )
            self.trail.append(center)

            if is_opentrack:
                self.position_var.set(
                    f"OpenTrack | "
                    f"X {result.source_x:+.2f} | "
                    f"Y {result.source_y:+.2f} | "
                    f"Z {result.source_z:+.2f} | "
                    f"Yaw {result.yaw_deg:+.1f}° | "
                    f"Pitch {result.pitch_deg:+.1f}° | "
                    f"Roll {result.roll_deg:+.1f}°"
                )
            else:
                id_text = (
                    "Face"
                    if self.solution_var.get() == "MediaPipe Face"
                    else f"ID {result.marker_id}"
                )

                if (
                    result.yaw_deg is not None
                    and result.pitch_deg is not None
                    and result.roll_deg is not None
                ):
                    pose_text = (
                        f" | Yaw {result.yaw_deg:+.1f}°"
                        f" | Pitch {result.pitch_deg:+.1f}°"
                        f" | Roll {result.roll_deg:+.1f}°"
                    )
                else:
                    pose_text = ""

                self.position_var.set(
                    f"{id_text} | "
                    f"X {result.x_px:+.0f}px | "
                    f"Y {result.y_px:+.0f}px | "
                    f"Angle {result.angle_deg:+.1f}°"
                    f"{pose_text}"
                )
        else:
            if is_opentrack:
                self.position_var.set("Waiting for OpenTrack UDP data...")
            else:
                self.position_var.set("Target not detected")

        draw_tracking_overlay(frame, result, self.trail)

        if is_opentrack and result is not None:
            cv2.putText(
                frame,
                f"Raw XYZ: {result.source_x:+.2f}, {result.source_y:+.2f}, {result.source_z:+.2f}",
                (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        image.thumbnail((1000, 620))

        tk_image = ImageTk.PhotoImage(image=image)

        self.video_label.configure(image=tk_image)
        self.video_label.image = tk_image

        # Tkinter remains responsive because each iteration returns immediately.
        self.root.after(15, self._update_frame)

    def on_close(self):
        # Cleanly release resources before destroying the Tk window.
        self.stop_tracking()
        self.root.destroy()


def main():
    # Launch the desktop application.
    root = tk.Tk()
    VisualTrackingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

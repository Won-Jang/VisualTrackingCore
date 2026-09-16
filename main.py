# Tkinter entry point and main UI for the VisualTrackingCore proof of concept.

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from collections import deque

import cv2
from PIL import Image, ImageTk

from adapters import adapter_names, create_adapter
from camera import Camera
from core import ResponseResolver, TrackingEngine
from core.camera_calibration import CameraCalibration
from core.planar_marker_pose import ORIENTATION_CONVENTION
from outputs import LSLOutputAdapter, LSLResponseOutputAdapter
from overlay import draw_tracking_overlay


class VisualTrackingApp:
    # Main desktop UI that selects, runs, and visualizes tracking adapters.

    def __init__(self, root):
        self.root = root
        self.root.title("VisualTrackingCore V0.9.0")
        self.root.geometry("1100x900")

        self.camera = Camera()
        self.adapter = None
        self.engine = TrackingEngine()
        self.running = False
        self._after_id = None
        self.trail = deque(maxlen=50)

        self.solution_var = tk.StringVar(value=adapter_names()[0])
        self.camera_var = tk.StringVar(value="0")
        self.target_id_var = tk.StringVar(value="0")
        self.mode_var = tk.StringVar(value="2D Marker Rotation")
        self.marker_size_var = tk.StringVar(value="50.0")
        self.camera_calibration_path_var = tk.StringVar()
        self.camera_calibration_status_var = tk.StringVar(value="Not required in 2D mode")
        self.response_source_var = tk.StringVar(value="Response source: Marker rotation")
        self.lsl_enabled_var = tk.BooleanVar(value=True)
        self.lsl_stream_name_var = tk.StringVar(value="VisualTrackingCore_Tracking")
        self.lsl_response_stream_name_var = tk.StringVar(value="VisualTrackingCore_Response")

        self.status_var = tk.StringVar(value="Stopped")
        self.position_var = tk.StringVar(value="No detection")
        self.response_var = tk.StringVar(value="Response: not calibrated")
        self.calibration_var = tk.StringVar(value="Reference: not set")

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

        self.marker_controls = ttk.LabelFrame(self.root, text="Marker tracking", padding=10)
        self.marker_controls.pack(fill="x", padx=10, pady=(0, 6))
        ttk.Label(self.marker_controls, text="Tracking Mode:").grid(row=0, column=0, sticky="w")
        self.mode_combo = ttk.Combobox(self.marker_controls, textvariable=self.mode_var,
                                      values=("2D Marker Rotation", "3D Pose"),
                                      state="readonly", width=22)
        self.mode_combo.grid(row=0, column=1, padx=5, sticky="w")
        self.mode_combo.bind("<<ComboboxSelected>>", self._on_solution_changed)
        ttk.Label(self.marker_controls, textvariable=self.response_source_var).grid(
            row=0, column=2, padx=10, sticky="w")
        self.pose_controls = ttk.Frame(self.marker_controls)
        self.pose_controls.grid(row=1, column=0, columnspan=4, sticky="w", pady=(6, 0))
        ttk.Label(self.pose_controls, text="Tag size (mm):").grid(row=0, column=0)
        self.marker_size_entry = ttk.Entry(self.pose_controls, textvariable=self.marker_size_var, width=8)
        self.marker_size_entry.grid(row=0, column=1, padx=5)
        ttk.Label(self.pose_controls, text="Calibration:").grid(row=0, column=2)
        self.camera_calibration_entry = ttk.Entry(
            self.pose_controls, textvariable=self.camera_calibration_path_var, width=47)
        self.camera_calibration_entry.grid(row=0, column=3, padx=5)
        self.camera_calibration_entry.bind("<FocusOut>", self._refresh_calibration_status)
        self.camera_calibration_entry.bind("<Return>", self._refresh_calibration_status)
        self.browse_calibration_button = ttk.Button(self.pose_controls, text="Browse...",
                                                    command=self._browse_calibration)
        self.browse_calibration_button.grid(row=0, column=4)
        ttk.Label(self.pose_controls, textvariable=self.camera_calibration_status_var).grid(
            row=1, column=0, columnspan=5, sticky="w", pady=3)
        ttk.Label(self.pose_controls, text="Overhead camera; flat head-top marker, printed top toward participant's front.").grid(
            row=2, column=0, columnspan=5, sticky="w")

        output_controls = ttk.LabelFrame(
            self.root,
            text="Output",
            padding=(10, 6),
        )
        output_controls.pack(fill="x", padx=10, pady=(0, 6))

        self.lsl_check = ttk.Checkbutton(
            output_controls,
            text="LSL",
            variable=self.lsl_enabled_var,
        )
        self.lsl_check.grid(row=0, column=0, padx=(0, 12), pady=2, sticky="w")

        ttk.Label(output_controls, text="Stream name:").grid(
            row=0, column=1, padx=(0, 5), pady=2, sticky="w"
        )
        self.lsl_stream_entry = ttk.Entry(
            output_controls,
            textvariable=self.lsl_stream_name_var,
            width=28,
        )
        self.lsl_stream_entry.grid(row=0, column=2, padx=5, pady=2, sticky="w")

        ttk.Label(
            output_controls,
            text="Tracking stream: X/Y/Z + orientation + status",
        ).grid(row=0, column=3, padx=12, pady=2, sticky="w")

        ttk.Label(output_controls, text="Response stream:").grid(
            row=1, column=1, padx=(0, 5), pady=2, sticky="w"
        )
        self.lsl_response_stream_entry = ttk.Entry(
            output_controls,
            textvariable=self.lsl_response_stream_name_var,
            width=28,
        )
        self.lsl_response_stream_entry.grid(row=1, column=2, padx=5, pady=2, sticky="w")
        ttk.Label(
            output_controls,
            text="Azimuth/Elevation + confidence + response valid",
        ).grid(row=1, column=3, padx=12, pady=2, sticky="w")

        calibration_controls = ttk.LabelFrame(
            self.root,
            text="Localization Response Calibration",
            padding=(10, 6),
        )
        calibration_controls.pack(fill="x", padx=10, pady=(0, 6))

        self.calibrate_button = ttk.Button(
            calibration_controls,
            text="Set 0° Reference",
            command=self.calibrate_reference,
            state="disabled",
        )
        self.calibrate_button.grid(row=0, column=0, padx=(0, 10), pady=2)
        ttk.Label(calibration_controls, textvariable=self.calibration_var).grid(
            row=0, column=1, padx=5, pady=2, sticky="w"
        )
        ttk.Label(calibration_controls, text="   |   ").grid(row=0, column=2)
        ttk.Label(calibration_controls, textvariable=self.response_var).grid(
            row=0, column=3, padx=5, pady=2, sticky="w"
        )

        info = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        info.pack(fill="x")

        ttk.Label(info, textvariable=self.status_var, wraplength=1040).pack(anchor="w")
        ttk.Label(info, textvariable=self.position_var, wraplength=1040).pack(anchor="w")

        self.video_label = ttk.Label(self.root)
        self.video_label.pack(fill="both", expand=True, padx=10, pady=10)

        self._on_solution_changed()

    def _on_solution_changed(self, _event=None):
        # Show only the controls relevant to the selected adapter.
        solution = self.solution_var.get()
        is_mediapipe = solution == "MediaPipe Face"

        self.input_label.config(text="Camera index:")
        self.camera_entry.grid()

        if is_mediapipe:
            self.target_id_entry.config(state="disabled")
            self.target_id_label.config(text="Target ID: (N/A)")
        else:
            self.target_id_entry.config(state="normal")
            self.target_id_label.config(text="Target ID:")
        self.mode_combo.config(state="disabled" if is_mediapipe else "readonly")
        if self._tracking_mode() == "3d_pose":
            self.pose_controls.grid()
            self.response_source_var.set("Response source: 3D yaw (overhead)")
            self._refresh_calibration_status()
        else:
            self.pose_controls.grid_remove()
            self.response_source_var.set("Response source: Yaw" if is_mediapipe
                                         else "Response source: Marker rotation")

    def _tracking_mode(self):
        if self.solution_var.get() in ("AprilTag", "ArUco") and self.mode_var.get() == "3D Pose":
            return "3d_pose"
        return "2d_rotation"

    def _browse_calibration(self):
        path = filedialog.askopenfilename(title="Select camera calibration",
                                         filetypes=[("NumPy calibration", "*.npz")])
        if path:
            self.camera_calibration_path_var.set(path)
            self._refresh_calibration_status()

    def _refresh_calibration_status(self, _event=None):
        try:
            calibration = CameraCalibration.load(self.camera_calibration_path_var.get())
            self.camera_calibration_status_var.set(
                f"Valid file | {calibration.image_width} x {calibration.image_height} | "
                f"RMS: {calibration.reprojection_error:.3f} px | Camera checked at Start")
        except ValueError:
            self.camera_calibration_status_var.set("Invalid/missing calibration; select a valid .npz file.")

    def _parse_target_id(self):
        # Parse a numeric marker ID; blank or ``all`` means no filtering.
        text = self.target_id_var.get().strip()

        if text == "" or text.lower() == "all":
            return None

        return int(text)

    @staticmethod
    def _response_resolver_for_solution(solution, tracking_mode="2d_rotation"):
        # Overhead paper-marker trackers use in-plane marker rotation as
        # horizontal response orientation. MediaPipe provides yaw directly.
        if solution in ("AprilTag", "ArUco") and tracking_mode == "2d_rotation":
            axis = "roll"
            orientation_label = "Marker rotation"
        else:
            axis = "yaw"
            orientation_label = "3D yaw (overhead)" if tracking_mode == "3d_pose" else "Yaw"

        return ResponseResolver(
            response_method="head_orientation",
            orientation_axis=axis,
            sign=1.0,
            orientation_label=orientation_label,
        )

    def calibrate_reference(self):
        # Define the participant's current orientation as speaker-array 0 degrees.
        try:
            reference = self.engine.calibrate_current()
        except Exception as exc:
            messagebox.showerror("Calibration failed", str(exc))
            return

        orientation_label = self.engine.response_resolver.orientation_label
        self.calibration_var.set(
            f"Reference: {orientation_label} {reference:+.1f}° = speaker-array 0°"
        )

    def start_tracking(self):
        # Create the selected adapter and start the UI update loop.
        if self.running:
            return

        solution = self.solution_var.get()
        tracking_mode = self._tracking_mode()

        try:
            if solution == "MediaPipe Face":
                target_id = 0
            else:
                target_id = self._parse_target_id()

            # All current adapters are camera-based and process frames captured
            # by VisualTrackingCore.
            camera_index = int(self.camera_var.get())
            calibration = None
            adapter_options = {}
            if solution in ("AprilTag", "ArUco"):
                adapter_options["tracking_mode"] = tracking_mode
            if tracking_mode == "3d_pose":
                calibration = CameraCalibration.load(self.camera_calibration_path_var.get())
                adapter_options.update(calibration=calibration,
                                       marker_size_mm=float(self.marker_size_var.get()))
            self.adapter = create_adapter(
                solution,
                target_id=target_id,
                **adapter_options,
            )
            self.camera.open(camera_index)
            if calibration is not None:
                self.camera.capture.set(cv2.CAP_PROP_FRAME_WIDTH, calibration.image_width)
                self.camera.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, calibration.image_height)
                ok, frame = self.camera.read()
                if not ok:
                    raise RuntimeError("Cannot read a frame to verify calibration resolution.")
                self.adapter.validate_frame(frame)
                self.camera_calibration_status_var.set(
                    f"Valid | Camera {frame.shape[1]} x {frame.shape[0]} | "
                    f"RMS: {calibration.reprojection_error:.3f} px")
            input_text = f"Camera {camera_index}"

            self.engine = TrackingEngine()
            self.engine.configure_input(self.adapter)
            self.engine.configure_response_resolver(
                self._response_resolver_for_solution(solution, tracking_mode)
            )

            output_names = []
            if self.lsl_enabled_var.get():
                tracking_output = LSLOutputAdapter(
                    stream_name=self.lsl_stream_name_var.get(),
                    source_name=self.adapter.name,
                    position_unit=self.adapter.position_unit,
                    metadata=self._stream_metadata(tracking_mode, calibration),
                )
                response_output = LSLResponseOutputAdapter(
                    stream_name=self.lsl_response_stream_name_var.get(),
                    source_name=self.adapter.name,
                    response_method="head_orientation",
                    metadata=self._stream_metadata(tracking_mode, calibration),
                )
                self.engine.add_tracking_output(tracking_output)
                self.engine.add_response_output(response_output)
                output_names.extend(["LSL Tracking", "LSL Response"])

            self.engine.start()
            output_text = ", ".join(output_names) if output_names else "None"

        except Exception as exc:
            if getattr(self, "engine", None) is not None:
                self.engine.stop()

            if self.adapter is not None:
                self.adapter.close()
                self.adapter = None

            self.camera.close()
            messagebox.showerror("Start failed", str(exc))
            return

        self.running = True
        self.trail.clear()

        self.status_var.set(
            f"Running: {solution} / {input_text} / Output: {output_text}"
        )
        self.calibration_var.set("Reference: not set")
        self.response_var.set("Response: not calibrated")

        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.solution_combo.config(state="disabled")
        self.lsl_check.config(state="disabled")
        self.lsl_stream_entry.config(state="disabled")
        self.lsl_response_stream_entry.config(state="disabled")
        self.calibrate_button.config(state="normal")
        for widget in (self.mode_combo, self.marker_size_entry, self.camera_calibration_entry,
                       self.browse_calibration_button, self.camera_entry, self.target_id_entry):
            widget.config(state="disabled")

        self._update_frame()

    def stop_tracking(self):
        # Stop tracking and release camera/adapter resources.
        self.running = False
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

        if getattr(self, "engine", None) is not None:
            self.engine.stop()

        self.camera.close()

        if self.adapter is not None:
            self.adapter.close()
            self.adapter = None

        self.status_var.set("Stopped")
        self.position_var.set("No detection")
        self.response_var.set("Response: not calibrated")
        self.calibration_var.set("Reference: not set")

        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.solution_combo.config(state="readonly")
        self.lsl_check.config(state="normal")
        self.lsl_stream_entry.config(state="normal")
        self.lsl_response_stream_entry.config(state="normal")
        self.calibrate_button.config(state="disabled")
        for widget in (self.marker_size_entry, self.camera_calibration_entry,
                       self.browse_calibration_button, self.camera_entry):
            widget.config(state="normal")
        self._on_solution_changed()

    def _stream_metadata(self, mode, calibration):
        metadata = {"tracking_mode": mode if self.solution_var.get() != "MediaPipe Face" else "face_pose",
                    "orientation_convention": ORIENTATION_CONVENTION if mode == "3d_pose" else "legacy",
                    "response_source": self.engine.response_resolver.orientation_label,
                    "azimuth_sign": "1", "confidence_semantics": "detection_validity_not_probability"}
        if calibration is not None:
            metadata.update(marker_size_mm=self.marker_size_var.get(),
                            camera_coordinates="x_right_y_down_z_away_mm",
                            calibration_resolution=f"{calibration.image_width}x{calibration.image_height}",
                            calibration_reprojection_error_px=str(calibration.reprojection_error),
                            calibration_timestamp=calibration.timestamp)
        return metadata

    def _update_frame(self):
        # Acquire/detect/draw one frame and schedule the next UI refresh.
        self._after_id = None
        if not self.running:
            return

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

        # Normalize all results and publish them through enabled output adapters.
        # The UI still visualizes only one primary result at a time.
        try:
            _frames, responses = self.engine.process_results(results)
        except Exception as exc:
            self.stop_tracking()
            messagebox.showerror("Output error", str(exc))
            return


        response = responses[0] if responses else None
        if response is not None and response.response_valid:
            self.response_var.set(
                f"Response azimuth: {response.response_azimuth_deg:+.1f}°"
            )
        elif self.engine.response_resolver and self.engine.response_resolver.calibrated:
            self.response_var.set("Response: tracking invalid")
        else:
            self.response_var.set("Response: not calibrated")

        result = results[0] if results else None

        if result is not None:
            center = (
                int(result.center_px[0]),
                int(result.center_px[1]),
            )
            self.trail.append(center)

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

            if self.solution_var.get() in ("AprilTag", "ArUco"):
                orientation_text = (
                    f" | Marker rotation {result.angle_deg:+.1f}°"
                )
            else:
                orientation_text = ""

            if self._tracking_mode() == "3d_pose":
                if result.pose_valid:
                    orientation_text = (f" | XYZ {result.source_x:+.1f}, {result.source_y:+.1f}, "
                                        f"{result.source_z:+.1f} mm")
                else:
                    orientation_text = " | 3D pose invalid"

            self.position_var.set(
                f"{id_text} | "
                f"X {result.x_px:+.0f}px | "
                f"Y {result.y_px:+.0f}px"
                f"{orientation_text}"
                f"{pose_text}"
            )
        else:
            self.position_var.set("Target not detected")

        marker_rotation_label = (
            "Marker rotation"
            if self.solution_var.get() in ("AprilTag", "ArUco") and self._tracking_mode() == "2d_rotation"
            else None
        )
        draw_tracking_overlay(
            frame,
            result,
            self.trail,
            orientation_label=marker_rotation_label,
        )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        image.thumbnail((1000, 620))

        tk_image = ImageTk.PhotoImage(image=image)

        self.video_label.configure(image=tk_image)
        self.video_label.image = tk_image

        # Tkinter remains responsive because each iteration returns immediately.
        self._after_id = self.root.after(15, self._update_frame)

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

# VisualTrackingCore V0.9.0

VisualTrackingCore is a modular localization-response framework for spatial-hearing experiments. Different tracking technologies can be used to observe a participant, while the framework converts the observation into a common response direction relative to the speaker array.

## V0.9.0: calibrated 3D marker tracking

AprilTag and ArUco now offer **2D Marker Rotation** and **3D Pose**. Existing
2D tracking, MediaPipe, zero-reference calibration, and the two LSL channel
layouts remain available. 3D mode is implemented for an **overhead camera
pointing vertically down at a rigid, flat marker on top of the head**.

| Mode | Requirements | Raw position | Response source |
|---|---|---|---|
| 2D Marker Rotation | Visible marker; no camera calibration | Image-center X/Y in pixels | Existing image-plane marker rotation |
| 3D Pose | Camera calibration and measured marker size | Camera X/Y/Z in mm | 3D overhead yaw |
| MediaPipe Face | Existing face model | Existing image coordinates | Existing front-camera yaw |

The 3D estimator uses four decoded marker corners, physical size, camera
intrinsics, distortion coefficients, and OpenCV PnP. Both marker adapters share
`core/planar_marker_pose.py`; speaker-array zeroing stays in `ResponseResolver`.
AprilTag keeps `tagStandard41h12`, ArUco keeps `DICT_4X4_50`, and both default to ID 0.

### Run

From the project root, activate the existing environment, then:

```powershell
python -m pip install -r requirements.txt
python main.py
```

For a new environment, use Python 3.13, `python -m venv .venv`, then activate
with `.\.venv\Scripts\Activate.ps1` on Windows or `source .venv/bin/activate`
on macOS. OpenCV 4.8+ is required for the ChArUco API; this milestone was tested
with OpenCV 5.0.0 on Windows. Camera/backend behavior on macOS needs hardware testing.

### Calibrate the camera

1. Generate a ChArUco board:

   ```powershell
   python tools/calibrate_camera.py --generate-board calibration/charuco_board.png
   ```

   Defaults: 5 x 7 squares, 30 mm square side, 22 mm marker side, `DICT_4X4_50`.
   The board is 150 x 210 mm, with a 10 mm margin on each side (170 x 230 mm
   total). PNG resolution metadata is 300 DPI. Print at **100% / actual size**,
   disable fit-to-page, and measure the squares. Mount on a flat, rigid surface.

2. Stop tracking in the main application, then open calibration capture:

   ```powershell
   python tools/calibrate_camera.py --camera 0 --width 1280 --height 720 --output calibration/webcam_calibration.npz
   ```

   Width/height are optional and must be supplied together. The tool records
   the actual resolution returned by the camera. If you change the board,
   supply matching `--squares-x`, `--squares-y`, `--square-mm`, and `--marker-mm`
   options to both generation and capture commands.

3. Move the board around the image, with varied tilts and distances. Press
   **Space** for each capture. Capture 15–25 useful views covering the image
   center and edges. Nearly duplicate views and insufficient/collinear corner
   sets are rejected. Press **C** after at least 15 views to calibrate and save;
   **Q/Esc** cancels. A low reprojection error alone does not establish accuracy.

4. Record the reported frame count, resolution, and reprojection RMS in pixels.
   The `.npz` includes `camera_matrix`, `dist_coeffs`, `image_width`,
   `image_height`, `reprojection_error`, UTC `timestamp`, and JSON
   `board_definition` (including OpenCV version and frame count). No pickled
   objects are accepted. Saving to an existing output path replaces that file.

Use the same physical camera, lens, focus, zoom/crop, and resolution when
tracking. Recalibrate if those change. Files are validated numerically; a valid
file does not prove it belongs to the connected camera. 3D mode requests the
calibrated resolution, checks the actual frame before starting LSL, and stops
on a later resolution mismatch. It never silently scales the calibration.

### Track a head in 3D

1. Mount the marker rigidly and flat on top of the head, with its **printed
   top toward the participant's front**. Keep the overhead camera vertical;
   do not use this convention with a front-facing or oblique camera.
2. Select AprilTag or ArUco and a single target ID; select **3D Pose**.
3. Enter the measured **detected-square side length in mm** and browse to the
   camera `.npz`. `50.0` is an example default, not a measurement of your print.
   For ArUco, measure the outer black square. For AprilTag, measure between
   the detector's corner locations; the full image/card/SVG size may include
   extra border cells, especially for `tagStandard41h12`. Do not automatically
   use the SVG page size from `tag_to_svg.py` as the pose marker size.
4. Click **Start**. Confirm **Response source: 3D yaw (overhead)** and valid
   XYZ in mm plus yaw/pitch/roll. Face the center speaker and click
   **Set 0° Reference**. Recalibrate after moving the camera or marker mount.
5. Check known left/right angles before recording participants. Camera roll
   within the horizontal plane is removed by zeroing; camera tilt is not.

For a 2D bench test, select **2D Marker Rotation**, start, and set the zero
reference. Camera calibration and marker size are not needed in this mode.

### Coordinate and response conventions

- Raw 3D translation is marker-center position in **camera coordinates**:
  X right, Y down in the image, Z away from the camera, all in millimeters.
- Marker axes are X toward printed right, Y toward printed top, Z outward
  from the printed face. Its decoded corner order is TL, TR, BR, BL.
- Overhead orientation uses `R = Rz(yaw) Rx(pitch) Ry(roll) R0`, where
  `R0 = diag(1, -1, -1)` is a flat face-on marker. Yaw follows the horizontal
  projection of the marker's forward axis, positive clockwise in the image;
  with the documented mounting, participant-right turns are positive. Pitch
  and roll are the other two angles of this explicitly defined sequence.
- This overhead yaw is rotation about camera Z. MediaPipe retains its existing
  front-camera convention. The LSL metadata identifies the marker convention;
  do not interpret all adapters' raw Euler angles as the same camera axes.
- `response_azimuth_deg = wrap(current_yaw - reference_yaw)` in 3D mode,
  in `[-180, 180)`. Zeroing binds to the source and marker ID. Another marker
  requires its own zero reference. Elevation remains unavailable (`NaN` in LSL).
- 2D image-plane rotation retains its legacy `roll` channel for compatibility;
  it is not a measured 3D roll. The active response source is shown in the UI.

### Outputs, validity, and limits

The tracking stream remains `x, y, z, yaw, pitch, roll, confidence,
tracking_valid, target_id`. The response stream remains `response_azimuth,
response_elevation, confidence, response_valid, target_id`. Metadata now includes
mode, orientation convention, response source, and 3D calibration/size details;
position units are correctly marked `mm` in 3D mode and `px` in 2D mode.

Lost markers, nonfinite geometry, failed pose estimation, points behind the
camera, undefined heading, or reprojection RMS above 3 px produce invalid
tracking/response samples with unavailable values, not valid zero poses. A
detected marker can remain outlined while its pose is invalid. Confidence is
still a detection-validity indicator, **not a calibrated probability**.

A single planar marker can have ambiguous/noisy pose, especially near frontal
views, at distance, or near occlusion. The estimator chooses a physically valid
candidate with the lowest reprojection error; temporal disambiguation is not
implemented. Tracking/response streams remain continuous, timestamped on LSL
send. Trial capture, stimulus control, auditory dosage, and localization-error
analysis remain outside this milestone.

### Verification

```powershell
python -m unittest discover -s tests -v
```

Tests cover synthetic 3D rotations/translation/size, actual detector corner
ordering using marker images, 2D compatibility, calibration files and synthetic
calibration recovery, tracking loss, response zeroing, and LSL schema/metadata.
These checks do not establish physical head-tracking accuracy. Bench-test 0°,
±30°, ±60°, and ±90° with pitch/roll variations, then check repeatability after
remounting and at the second site. Confirm live LSL reception before data collection.

Implementation references: [OpenCV PnP](https://docs.opencv.org/4.13.0/d5/d1f/calib3d_solvePnP.html),
[ChArUco calibration](https://docs.opencv.org/4.13.0/da/d13/tutorial_aruco_calibration.html),
and [AprilTag corner construction](https://github.com/AprilRobotics/apriltag/blob/master/apriltag.c).


[AprilTag tracking]
![AprilTag TestBench](docs/images/setup.png)
![AprilTag 0 Degree](docs/images/0.png)
![AprilTag -45 Degree](docs/images/-45.png)
![AprilTag 45 Degree](docs/images/45.png)
![AprilTag -90 Degree](docs/images/-90.png)
![AprilTag 90 Degree](docs/images/90.png)


[ArUco tracking]
![ArUco tracking example](docs/images/aruco_tracking.png)

## ###In Development Below - FILL/FIX ME###

### Some Notes

- For our AprilTag/ArUco head-tracking setup, make the tag as large as you reasonably can while still fitting comfortably on the head/cap.
- Tag width ≈ 1/20 to 1/30 of the camera-to-tag distance

1. at 0.5 m distance → ~20–30 mm can work
2. at 1 m distance → ~40–60 mm is safer
3. at 1.5 m distance → ~60–80 mm is better

```text
AprilTag / ArUco test tag:
50 mm × 50 mm active tag area
~60–70 mm total printed card
```

### ArUco Marker Generator

`generate_aruco.py` creates a printable ArUco marker using the `DICT_4X4_50` dictionary.

Default:

```powershell
python generate_aruco.py
```

Generates:

```text
aruco_4x4_50_id0.png
```

Default values:

Marker ID: 0
Image size: 1000 × 1000 pixels

Optional parameters:

```powershell
python generate_aruco.py --id 5 --size 1500 --output my_marker.png
```

- `--id` marker ID
- `--size` image size in pixels
- `--output` output filename

The printed marker ID must match the ID selected in VisualTrackingCore.

### AprilTag-imgs
=============

Images of all tags from all the pre-generated [AprilTag 3](https://github.com/AprilRobotics/apriltags) families. You can generate your own layouts or images of tags using our other repo, [AprilTag-generation](https://github.com/AprilRobotics/apriltag-generation).

If the format of the markers is very small (ex : by default, 9x9 pixels), you'll need to rescale them. To do so, you may use the following imagemagick command (Unix) : 

~~~
convert <small_marker>.png -scale <scale_chosen_in_percent>% <big_marker>.png
~~~

Alternately, you can use the supplied native Python 3 script `tag_to_svg.py` to create a SVG (Scalable Vector Graphics) Version of a tag. For example:
~~~
python ./tools/tag_to_svg.py ./markers/tag41_12_00000.png ./markers/tag41_12_00000.svg --size=50mm
~~~

## Measurement model

V0.8.2 separates **sensor tracking data** from the **experimental localization response**.

```text
AprilTag / ArUco / MediaPipe / PointTracker
                         |
                         v
                  TrackingFrame
              raw normalized sensor pose
                         |
                         v
               ResponseResolver
              calibrated 0° reference
                         |
                         v
             LocalizationResponse
        speaker-array-relative response
                         |
                +--------+--------+
                |                 |
                v                 v
        LSL Tracking       LSL Response
```

The primary experiment-facing measurement is:

```text
response_azimuth_deg
```

`0°` is the participant's calibrated forward direction / speaker-array center. Elevation is included in the response model for future 3-D localization tasks but is not yet resolved by the current V0.8 calibration path.

## Two core data models

### TrackingFrame

Represents what the selected tracking technology measured.

```text
timestamp
source
target_id
tracking_valid
confidence
x / y / z
position_unit
yaw_deg / pitch_deg / roll_deg
```

Current adapter behavior:

| Adapter | Tracking orientation used for localization response |
|---|---|
| AprilTag | In-plane marker rotation (`roll_deg`) |
| ArUco | In-plane marker rotation (`roll_deg`) |
| MediaPipe Face | Estimated head `yaw_deg` |
| PointTracker | Planned |

For AprilTag and ArUco, this mapping assumes an overhead-camera arrangement where marker rotation in the image corresponds to horizontal head orientation. This should be verified experimentally for the final camera geometry.

### LocalizationResponse

Represents the experiment-facing response rather than raw tracker pose.

```text
timestamp
source
response_method
response_valid
confidence
response_azimuth_deg
response_elevation_deg
target_id
```

The response model intentionally does **not** contain the stimulus speaker ID, target angle, or localization error. Those are trial/experiment variables and should remain in the experimental software. VisualTrackingCore reports what direction the participant indicated.

## V0.8.2 terminology update

The UI now distinguishes the physical orientation measurement used by each tracker:

- **AprilTag / ArUco:** `Marker rotation` (2-D in-plane rotation in the camera image)
- **MediaPipe:** `Yaw`
- **Response azimuth:** the calibrated experiment-facing angle relative to speaker-array `0°`

This avoids describing the paper-marker measurement as generic `Angle` or full 3-D `Roll`.

## Calibration

While tracking is running:

1. Ask the participant to face the speaker-array center / intended `0°` direction.
2. Click **Set 0° Reference**.
3. VisualTrackingCore stores the current orientation as the reference.
4. Subsequent orientations are converted to signed azimuth relative to that reference.

Example:

```text
Calibration yaw:       +13.2°
Current yaw:           +48.5°
Response azimuth:      +35.3°
```

Angles are wrapped to the range `[-180°, 180°)`.

## LSL outputs

When LSL is enabled, V0.8.2 creates two independent streams.

### VisualTrackingCore_Tracking

Raw normalized tracking stream:

```text
x
y
z
yaw
pitch
roll
confidence
tracking_valid
target_id
```

Unavailable values are `NaN`. Tracking loss is explicit with `tracking_valid = 0`.

### VisualTrackingCore_Response

Experiment-facing response stream:

```text
response_azimuth
response_elevation
confidence
response_valid
target_id
```

Before calibration, or when the tracker is lost, `response_valid = 0` and unavailable response angles are `NaN`.

The response stream can therefore be consumed by MATLAB/Python experimental code without needing to understand whether the source was AprilTag, ArUco, MediaPipe, or a future PointTracker adapter.

## Project structure

```text
VisualTrackingCore/
├── main.py
├── camera.py
├── tracking_types.py
├── adapters/
│   ├── base.py
│   ├── apriltag_adapter.py
│   ├── aruco_adapter.py
│   ├── mediapipe_adapter.py
│   └── registry.py
├── core/
│   ├── tracking_frame.py
│   ├── localization_response.py
│   ├── response_resolver.py
│   └── tracking_engine.py
├── outputs/
│   ├── base.py
│   ├── lsl_output.py
│   └── lsl_response_output.py
├── models/
├── markers/
├── tools/
└── tests/
```

## Setup

From PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

From Bash:

```Bash
brew install python@3.13
brew install python-tk@3.13
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

## Basic V0.8 test procedure

1. Start `main.py`.
2. Select a tracker.
3. Keep LSL enabled if LabRecorder/another LSL client will be used.
4. Click **Start**.
5. Confirm raw tracking values update.
6. Face the center speaker / reference direction.
7. Click **Set 0° Reference**.
8. Rotate toward known speaker angles.
9. Compare `Response azimuth` with the known speaker positions.
10. Confirm both `VisualTrackingCore_Tracking` and `VisualTrackingCore_Response` appear in the LSL client.

## V0.8.0 changes

- Added `LocalizationResponse` experiment-facing data model.
- Added `ResponseResolver` calibration layer.
- Added speaker-array-relative `response_azimuth_deg`.
- Added angle wrapping across ±180°.
- Added **Set 0° Reference** control to the UI.
- Added live response-azimuth display.
- Split LSL into raw Tracking and calibrated Response streams.
- Preserved V0.7 tracker adapters and raw tracking behavior.
- Added tests for calibration, invalid tracking, response routing, and angle wrapping.

## Next development steps

1. Bench-test known angles (for example 0°, ±30°, ±60°, ±90°) for each current adapter.
2. Confirm/sign-correct the physical left/right azimuth convention for the actual camera/speaker geometry.
3. Add response capture/confirmation semantics (continuous vs participant/experimenter capture event).
4. Add PointTracker as another input adapter.
5. Add optional speaker-array configuration and validation tools without coupling stimulus presentation to the tracking core.
6. Add other output adapters only when required.

##


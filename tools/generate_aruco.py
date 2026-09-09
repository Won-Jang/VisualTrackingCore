# Utility script for generating a printable OpenCV ArUco marker image.
#
# This script creates a PNG image for a marker from the same dictionary used by
# the default ArUco adapter in VisualTrackingCore.
#
# Default behavior
# ----------------
# If the script is run with no command-line arguments:
#
#     python generate_aruco.py
#
# it will generate:
#   - dictionary: DICT_4X4_50
#   - marker ID:  0
#   - image size: 1000 x 1000 pixels
#   - output:     aruco_4x4_50_id0.png
#
# Optional command-line arguments
# -------------------------------
# You can also provide:
#   --id       Marker ID to generate
#   --size     Output image size in pixels
#   --output   Output filename
#
# Examples:
#   python generate_aruco.py --id 5
#   python generate_aruco.py --id 5 --size 1500
#   python generate_aruco.py --id 5 --size 1500 --output my_marker.png
#
# Important note
# --------------
# The "size" argument controls the PNG resolution in pixels, NOT the physical
# print size in millimeters. Physical print size is chosen later when printing.

import argparse
import cv2


# Use the same dictionary configured by the default ArUco adapter.
DICTIONARY_ID = cv2.aruco.DICT_4X4_50
DICTIONARY_NAME = "DICT_4X4_50"

# Default values used when no command-line parameters are provided.
DEFAULT_MARKER_ID = 0
DEFAULT_SIZE = 1000


def build_parser() -> argparse.ArgumentParser:
    """
    Create the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Parser configured with optional marker-generation arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate a printable ArUco marker PNG using the "
            "VisualTrackingCore default dictionary."
        )
    )

    parser.add_argument(
        "--id",
        type=int,
        default=DEFAULT_MARKER_ID,
        help=(
            f"Marker ID to generate "
            f"(default: {DEFAULT_MARKER_ID})"
        ),
    )

    parser.add_argument(
        "--size",
        type=int,
        default=DEFAULT_SIZE,
        help=(
            "Output image size in pixels "
            f"(default: {DEFAULT_SIZE})"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output PNG filename "
            "(default: aruco_4x4_50_id<ID>.png)"
        ),
    )

    return parser


def main() -> None:
    """
    Generate and save the requested ArUco marker image.
    """
    parser = build_parser()
    args = parser.parse_args()

    # Load the predefined dictionary used by the project.
    dictionary = cv2.aruco.getPredefinedDictionary(DICTIONARY_ID)

    # Validate the marker size.
    if args.size <= 0:
        raise ValueError("--size must be a positive integer.")

    # Validate marker ID range for the selected dictionary.
    max_markers = int(dictionary.bytesList.shape[0])
    if not (0 <= args.id < max_markers):
        raise ValueError(
            f"--id must be between 0 and {max_markers - 1} "
            f"for {DICTIONARY_NAME}."
        )

    # If the user does not provide an output filename, generate one
    # automatically from the marker ID.
    output = args.output or f"aruco_4x4_50_id{args.id}.png"

    # Generate the marker image.
    marker = cv2.aruco.generateImageMarker(
        dictionary,
        args.id,
        args.size,
    )

    # Save the result to disk.
    ok = cv2.imwrite(output, marker)
    if not ok:
        raise RuntimeError(f"Failed to save marker image: {output}")

    print(f"Dictionary : {DICTIONARY_NAME}")
    print(f"Marker ID  : {args.id}")
    print(f"Image size : {args.size} x {args.size} pixels")
    print(f"Saved      : {output}")


if __name__ == "__main__":
    main()

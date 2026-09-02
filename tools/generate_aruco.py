# Utility script for generating a printable OpenCV ArUco marker image.

import cv2

# Use the same dictionary configured by the default ArUco adapter.
dictionary = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_4X4_50
)

# Marker ID 0 is the default POC target used by the UI.
marker_id = 0
size = 1000

marker = cv2.aruco.generateImageMarker(
    dictionary,
    marker_id,
    size
)

output = "aruco_4x4_50_id0.png"
cv2.imwrite(output, marker)

print(f"Saved {output}")

"""small script to save the pose of an aruco marker to a pickle file for later use.
"""

import pickle
from pathlib import Path

import cv2
from camera_toolkit.aruco import get_aruco_marker_poses
from camera_toolkit.zed2i import Zed2i
import pyzed.sl as sl
import cloth_manipulation.camera_mapping as cm


Zed2i.list_camera_serial_numbers()
zed = Zed2i(resolution=sl.RESOLUTION.HD2K, serial_number=cm.CameraMapping.serial_top)
img = zed.get_rgb_image()
img = zed.image_shape_torch_to_opencv(img)
cam_matrix = zed.get_camera_matrix()
print(img.shape)
img, t, r, _ = get_aruco_marker_poses(img, cam_matrix, 0.106, cv2.aruco.DICT_6X6_250, True)
print(t)
print(r)
cv2.imshow(",", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
with open(Path(__file__).parent / "marker.pickle", "wb") as f:
    pickle.dump([t[0], r[0]], f)
zed.close()

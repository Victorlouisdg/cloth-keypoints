from typing import List, Tuple
from camera_toolkit.zed2i import Zed2i
import cv2
import pyzed.sl as sl
import numpy as np
from pathlib import Path
import pickle
from camera_toolkit.reproject import reproject_to_world_z_plane
from camera_toolkit.aruco import get_aruco_marker_poses
import cloth_manipulation.camera_mapping as cm


def draw_center_circle(image) -> np.ndarray:
    h, w, _ = image.shape
    center_u = w // 2
    center_v = h // 2
    center = (center_u, center_v)
    image = cv2.circle(image, center, 1, (0, 255, 0), thickness=2)
    return image

# load camera to marker transform
with open(Path(__file__).parent.parent / "cloth_manipulation" / "marker.pickle", "rb") as f:
    aruco_in_camera_position, aruco_in_camera_orientation = pickle.load(f)
# get camera extrinsics transform
aruco_in_camera_transform = np.eye(4)
aruco_in_camera_transform[:3, :3] = aruco_in_camera_orientation
aruco_in_camera_transform[:3, 3] = aruco_in_camera_position

resolution = sl.RESOLUTION.HD720
zed = Zed2i(resolution=resolution, serial_number=cm.CameraMapping.serial_top)

def project_to_camera_plane(point_3D: List[float]) -> Tuple[int, int]:
    point_homogeneous = np.ones((4,1))
    point_homogeneous[:3, 0] = point_3D
    world_to_camera = aruco_in_camera_transform
    point_camera = world_to_camera @ point_homogeneous
    point_3D = point_camera[:3, 0]
    point_camera_2D = zed.get_camera_matrix() @ point_3D
    point_2D = point_camera_2D[:2] / point_camera_2D[2]
    # print(point_3D, point_camera_2D, point_2D)
    u, v = point_2D
    return int(u), int(v)

window_name = "GUI"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)

def clicked_callback_cv(event, x, y, flags, param):
    global u_clicked, v_clicked
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print(f"clicked on {x}, {y}")
        keypoints_in_world = reproject_to_world_z_plane(
            np.array([[x,y]]), zed.get_camera_matrix(), aruco_in_camera_transform
        )
        print(keypoints_in_world)

# mark the keypoints in image plane by clicking
cv2.setMouseCallback(window_name, clicked_callback_cv)

print(aruco_in_camera_transform)
print(zed.get_camera_matrix())

while True:
    image = zed.get_rgb_image()
    origin = project_to_camera_plane([0.0, 0.0, 0.0])

    image = zed.image_shape_torch_to_opencv(image)
    image = image.copy()

    cv2.drawFrameAxes(image, zed.get_camera_matrix(), np.zeros(4), np.identity(3), np.zeros(3), 0.05)
    image, t, r, _ = get_aruco_marker_poses(image, zed.get_camera_matrix(), 0.106, cv2.aruco.DICT_6X6_250, True)

    image = cv2.circle(image, origin, 10, (0, 255, 255), thickness=2)

    camera_look = [int(c) for c in zed.get_camera_matrix()[:2, -1].T]
    image = cv2.circle(image, camera_look, 2, (255, 0, 255), thickness=2)

    x_pos = project_to_camera_plane([1.0, 0.0, 0.0])
    x_neg = project_to_camera_plane([-0.5, 0.0, 0.0])
    y_pos = project_to_camera_plane([0.0, 1.0, 0.0])
    y_neg = project_to_camera_plane([0.0, -0.5, 0.0])
    image = cv2.line(image, x_pos, x_neg, color=(0,0,255), thickness=2)
    image = cv2.line(image, y_pos, y_neg, color=(0,255,0), thickness=2)

    image = draw_center_circle(image)

    cv2.imshow(window_name, image)
    key = cv2.waitKey(10)
    if key == ord("q"):
        cv2.destroyAllWindows()
        zed.close()
        break
 
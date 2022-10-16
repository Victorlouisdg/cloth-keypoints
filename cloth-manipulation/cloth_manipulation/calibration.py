import pickle
import time
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
from camera_toolkit.aruco import get_aruco_marker_poses
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.camera_mapping import CameraMapping
from cloth_manipulation.gui import Panel, draw_center_circle, draw_world_axes


@cache
def load_saved_calibration():
    with open(Path(__file__).parent / "marker.pickle", "rb") as f:
        aruco_in_camera_position, aruco_in_camera_orientation = pickle.load(f)
    # get camera extrinsics transform
    aruco_in_camera_transform = np.eye(4)
    aruco_in_camera_transform[:3, :3] = aruco_in_camera_orientation
    aruco_in_camera_transform[:3, 3] = aruco_in_camera_position
    return aruco_in_camera_transform


def save_calibration(rotation_matrix, translation):
    with open(Path(__file__).parent / "marker.pickle", "wb") as f:
        pickle.dump([translation, rotation_matrix], f)


if __name__ == "__main__":
    resolution = sl.RESOLUTION.HD720
    zed = Zed2i(resolution=resolution, serial_number=CameraMapping.serial_top, fps=30)

    # Configure custom project-wide InputTransform based on camera, resolution, etc.
    _, h, w = zed.get_rgb_image().shape
    panel = Panel(np.zeros((h, w, 3), dtype=np.uint8))

    print("Press s to save Marker pose, q to quit.")

    window_name = "GUI"
    cv2.namedWindow(window_name)

    while True:
        start_time = time.time()
        image = zed.get_rgb_image()
        image = zed.image_shape_torch_to_opencv(image)
        image = image.copy()
        cam_matrix = zed.get_camera_matrix()
        image, translations, rotations, _ = get_aruco_marker_poses(
            image, cam_matrix, 0.106, cv2.aruco.DICT_6X6_250, True
        )
        image = draw_center_circle(image)

        if rotations is not None:
            aruco_in_camera_transform = np.eye(4)
            aruco_in_camera_transform[:3, :3] = rotations[0]
            aruco_in_camera_transform[:3, 3] = translations[0]
            world_to_camera = aruco_in_camera_transform
            camera_matrix = zed.get_camera_matrix()
            image = draw_world_axes(image, world_to_camera, camera_matrix)

        panel.fill_image_buffer(image)

        cv2.imshow(window_name, panel.image_buffer)
        key = cv2.waitKey(10)
        if key == ord("s") and rotations is not None:
            print("Saving current marker pose to pickle.")
            save_calibration(rotations[0], translations[0])
            time.sleep(5)
        if key == ord("q"):
            cv2.destroyAllWindows()
            zed.close()
            break

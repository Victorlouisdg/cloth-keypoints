import argparse

import cv2
import numpy as np
import pyzed.sl as sl
from camera_toolkit.reproject import reproject_to_world_z_plane
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.calibration import load_saved_calibration
from cloth_manipulation.camera_mapping import CameraMapping
from cloth_manipulation.controllers import FoldTowelController, ReorientTowelController
from cloth_manipulation.hardware.setup_hardware import setup_victor_louise
from cloth_manipulation.manual_keypoints import get_manual_keypoints


def run_and_visualize(controller, victor_louise):
    resolution = sl.RESOLUTION.HD720
    zed = Zed2i(resolution=resolution, serial_number=CameraMapping.serial_top, fps=30)
    world_to_camera = load_saved_calibration()
    camera_matrix = zed.get_camera_matrix()

    victor_louise.dual_move_tcp(victor_louise.left.out_of_way_pose, victor_louise.right.out_of_way_pose)
    image = zed.get_rgb_image()

    image = Zed2i.image_shape_torch_to_opencv(image)
    image = image.copy()

    keypoints_in_camera = np.array(get_manual_keypoints(image))
    keypoints_in_world = reproject_to_world_z_plane(keypoints_in_camera, camera_matrix, world_to_camera)

    victor_louise.dual_move_tcp(victor_louise.left.home_pose, victor_louise.right.home_pose)

    image = controller.visualize_plan(image, keypoints_in_world, world_to_camera, camera_matrix)

    cv2.imshow("GUI", image)
    cv2.waitKey(1000)  # without this the image is black

    controller.act(keypoints_in_world)

    image_after = zed.get_rgb_image()
    image_after = Zed2i.image_shape_torch_to_opencv(image_after)
    image_after = image_after.copy()
    image_after = controller.visualize_plan(image_after, keypoints_in_world, world_to_camera, camera_matrix)

    image_before_after = np.zeros_like(image)
    h, w, _ = image.shape
    p = (w - h) // 2
    image_before_after[:, : w // 2, :] = image[:, p : p + w // 2, :]
    image_before_after[:, w // 2 :, :] = image_after[:, p : p + w // 2, :]

    cv2.imshow("GUI", image_before_after)
    cv2.waitKey(0)  # without this the image is black

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("controller")
    args = parser.parse_args()

    victor_louise = setup_victor_louise()

    controllers = {
        FoldTowelController.__name__: FoldTowelController(victor_louise),
        ReorientTowelController.__name__: ReorientTowelController(victor_louise),
    }
    controller = controllers[args.controller]
    run_and_visualize(controller, victor_louise)

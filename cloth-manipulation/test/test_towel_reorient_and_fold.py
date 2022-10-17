import time

import cv2
import numpy as np
import pyzed.sl as sl
from camera_toolkit.reproject import reproject_to_world_z_plane
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.calibration import load_saved_calibration
from cloth_manipulation.camera_mapping import CameraMapping
from cloth_manipulation.controllers import ReorientAndFoldTowelController
from cloth_manipulation.hardware.setup_hardware import setup_victor_louise
from cloth_manipulation.input_transform import InputTransform
from cloth_manipulation.observers import KeypointObserver

victor_louise = setup_victor_louise()
controller = ReorientAndFoldTowelController(victor_louise)
keypoint_observer = KeypointObserver()

resolution = sl.RESOLUTION.HD720
control_image_crop_size = 600

zed = Zed2i(resolution=resolution, serial_number=CameraMapping.serial_top, fps=30)

# Configure custom project-wide InputTransform based on camera, resolution, etc.
_, h, w = zed.get_rgb_image().shape
InputTransform.crop_start_u = (w - control_image_crop_size) // 2
InputTransform.crop_width = control_image_crop_size
InputTransform.crop_start_v = (h - control_image_crop_size) // 2 + 40
InputTransform.crop_height = control_image_crop_size
world_to_camera = load_saved_calibration()
camera_matrix = zed.get_camera_matrix()

victor_louise.dual_move_tcp(victor_louise.left.home_pose, victor_louise.right.home_pose)

# victor_louise.dual_move_tcp(victor_louise.left.out_of_way_pose, victor_louise.right.out_of_way_pose)

while not controller.finished:
    image = zed.get_rgb_image()

    keypoints = keypoint_observer.observe(image)
    keypoints_in_camera = InputTransform.reverse_transform_keypoints(np.array(keypoints))
    keypoints_in_world = reproject_to_world_z_plane(keypoints_in_camera, camera_matrix, world_to_camera)

    image = Zed2i.image_shape_torch_to_opencv(image)
    image = image.copy()

    image = controller.visualize_plan(image, keypoints_in_world, world_to_camera, camera_matrix)
    cv2.imshow("GUI", image)
    cv2.waitKey(100)  # without this the image is black
    controller.act(keypoints_in_world)

time.sleep(5)
cv2.destroyAllWindows()

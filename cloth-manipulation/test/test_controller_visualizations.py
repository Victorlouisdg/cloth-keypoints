import time
from collections import deque

import cv2
import numpy as np
import pyzed.sl as sl
from camera_toolkit.reproject import reproject_to_world_z_plane
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.calibration import load_saved_calibration
from cloth_manipulation.camera_mapping import CameraMapping
from cloth_manipulation.controllers import FoldTowelController, PickReorientTowelController, ReorientTowelController
from cloth_manipulation.gui import Panel, draw_cloth_transform_rectangle
from cloth_manipulation.hardware.setup_hardware import setup_fake_victor_louise
from cloth_manipulation.input_transform import InputTransform
from cloth_manipulation.observers import KeypointObserver

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

panel = Panel(np.zeros((h, w, 3), dtype=np.uint8))

print("Press q to quit.")

window_name = "GUI"
cv2.namedWindow(window_name)

loop_time_queue = deque(maxlen=15)
fps = -1

world_to_camera = load_saved_calibration()
camera_matrix = zed.get_camera_matrix()

victor_louise = setup_fake_victor_louise()
reorient_towel_controller = ReorientTowelController(victor_louise)
fold_towel_controller = FoldTowelController(victor_louise)
pick_reorient_towel_controller = PickReorientTowelController(victor_louise)

controllers = [reorient_towel_controller, fold_towel_controller, pick_reorient_towel_controller]
visualized_controller_index = 0


def mouse_callback(event, x, y, flags, parm):
    global visualized_controller_index

    if event == cv2.EVENT_LBUTTONDOWN:
        visualized_controller_index += 1
        visualized_controller_index %= len(controllers)


cv2.setMouseCallback(window_name, mouse_callback)

while True:
    start_time = time.time()
    image = zed.get_rgb_image()

    keypoints = keypoint_observer.observe(image)
    # visualization_image = keypoint_observer.visualize_last_observation()

    image = zed.image_shape_torch_to_opencv(image)
    image = image.copy()
    # insert_transformed_into_original(image, visualization_image)
    image = draw_cloth_transform_rectangle(image)

    if len(keypoints) == 4:
        controller = controllers[visualized_controller_index]
        keypoints_in_camera = InputTransform.reverse_transform_keypoints(np.array(keypoints))
        keypoints_in_world = reproject_to_world_z_plane(keypoints_in_camera, camera_matrix, world_to_camera)
        image = controller.visualize_plan(image, keypoints_in_world, world_to_camera, camera_matrix)

    panel.fill_image_buffer(image)
    cv2.putText(panel.image_buffer, f"fps: {fps:.1f}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow(window_name, panel.image_buffer)
    key = cv2.waitKey(10)
    if key == ord("q"):
        cv2.destroyAllWindows()
        zed.close()
        break

    end_time = time.time()
    loop_time = end_time - start_time
    loop_time_queue.append(loop_time)
    fps = 1.0 / np.mean(list(loop_time_queue))

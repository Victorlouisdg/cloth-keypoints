from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.calibration import load_saved_calibration
import cv2
import pyzed.sl as sl
import numpy as np
from cloth_manipulation.manual_keypoints import ClothTransform
import time
import cloth_manipulation.camera_mapping as cm
from cloth_manipulation.gui import Panel, draw_cloth_transform_rectangle, visualize_towel_reorient_pull
from cloth_manipulation.observers import KeypointObserver
from collections import deque
from cloth_manipulation.hardware.fake_hardware import FakeDualArm, FakeRobot
from cloth_manipulation.motion_primitives.pull import TowelReorientPull
from camera_toolkit.reproject import reproject_to_world_z_plane

keypoint_observer =  KeypointObserver()

resolution = sl.RESOLUTION.HD720
control_image_crop_size = 600

zed = Zed2i(resolution=resolution,serial_number=cm.CameraMapping.serial_top, fps=30)

# Configure custom project-wide ClothTransform based on camera, resolution, etc.
_, h ,w = zed.get_rgb_image().shape
ClothTransform.crop_start_u = (w - control_image_crop_size) // 2
ClothTransform.crop_width = control_image_crop_size
ClothTransform.crop_start_v = (h - control_image_crop_size) // 2
ClothTransform.crop_height = control_image_crop_size

panel = Panel(np.zeros((h, w, 3), dtype=np.uint8))

print("Press q to quit.")

window_name = "GUI"
cv2.namedWindow(window_name)

loop_time_queue = deque(maxlen=15)
fps = -1

world_to_camera = load_saved_calibration()
camera_matrix = zed.get_camera_matrix()

victor = FakeRobot(robot_in_world_position=[-0.4,0,0])
louise = FakeRobot(robot_in_world_position=[0.4,0,0])
dual_arm = FakeDualArm(victor, louise)

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
        keypoints_in_camera = ClothTransform.reverse_transform_keypoints(np.array(keypoints))
        keypoints_in_world = reproject_to_world_z_plane(keypoints_in_camera, camera_matrix, world_to_camera)
        pull = TowelReorientPull(keypoints_in_world, dual_arm)
        image = visualize_towel_reorient_pull(image, pull, world_to_camera, camera_matrix)

    panel.fill_image_buffer(image)
    cv2.putText(panel.image_buffer, f"fps: {fps:.1f}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

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

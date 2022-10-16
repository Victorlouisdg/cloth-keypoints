import time
from collections import deque

import cv2
import numpy as np
import pyzed.sl as sl
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.camera_mapping import CameraMapping
from cloth_manipulation.gui import Panel, draw_cloth_transform_rectangle, insert_transformed_into_original
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
InputTransform.crop_start_v = (h - control_image_crop_size) // 2
InputTransform.crop_height = control_image_crop_size

panel = Panel(np.zeros((h, w, 3), dtype=np.uint8))

print("Press q to quit.")

window_name = "GUI"
cv2.namedWindow(window_name)

loop_time_queue = deque(maxlen=15)
fps = -1

while True:
    start_time = time.time()
    image = zed.get_rgb_image()

    keypoints = keypoint_observer.observe(image)
    visualization_image = keypoint_observer.visualize_last_observation()

    image = zed.image_shape_torch_to_opencv(image)
    image = image.copy()
    insert_transformed_into_original(image, visualization_image)
    image = draw_cloth_transform_rectangle(image)

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

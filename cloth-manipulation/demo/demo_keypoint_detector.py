import threading
import time
from collections import deque

import cv2
import numpy as np
import pyzed.sl as sl
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.camera_mapping import CameraMapping
from cloth_manipulation.gui import FourPanels
from cloth_manipulation.input_transform import InputTransform
from cloth_manipulation.observers import KeypointObserver

keypoint_observer = KeypointObserver()

resolution = sl.RESOLUTION.HD720
control_image_crop_size = 600

low_resolution = False
if low_resolution:
    resolution = sl.RESOLUTION.VGA
    control_image_crop_size = 300

serial_numbers = [CameraMapping.serial_top, CameraMapping.serial_side, CameraMapping.serial_front]
zeds = {n: Zed2i(resolution=resolution, serial_number=n, fps=30) for n in serial_numbers}

# Configure custom project-wide InputTransform based on camera, resolution, etc.
_, h, w = zeds[CameraMapping.serial_top].get_rgb_image().shape
InputTransform.crop_start_u = (w - control_image_crop_size) // 2
InputTransform.crop_width = control_image_crop_size
InputTransform.crop_start_v = (h - control_image_crop_size) // 2
InputTransform.crop_height = control_image_crop_size

panels = FourPanels()

# Global vars for use in control thread
stop_control_thread = False
control_image_index = -1
control_image = None
control_visualization_panel = panels.top_left


def control_loop():
    global stop_control_thread
    global control_image_index
    global control_image
    global control_visualization_panel
    global keypoint_observer

    while not stop_control_thread:
        if control_image is None:
            time.sleep(0.1)
            print("No control image.")
            continue

        keypoint_observer.observe(control_image)
        control_visualization_image = keypoint_observer.visualize_last_observation()
        control_visualization_panel.fill_image_buffer(control_visualization_image)


control_thread = threading.Thread(target=control_loop)
control_thread.start()

print("Press q to quit.")

window_name = "GUI"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

loop_time_queue = deque(maxlen=15)
fps = -1

while True:
    start_time = time.time()
    images = {serial_number: zed.get_rgb_image() for serial_number, zed in zeds.items()}

    control_image = images[CameraMapping.serial_top].copy()
    control_image_index += 1

    for serial_number, image in images.items():
        images[serial_number] = Zed2i.image_shape_torch_to_opencv(image)

    panels.top_right.fill_image_buffer(images[CameraMapping.serial_top])
    panels.bottom_left.fill_image_buffer(images[CameraMapping.serial_front])
    panels.bottom_right.fill_image_buffer(images[CameraMapping.serial_side])

    cv2.putText(
        panels.image_buffer,
        f"camera fps: {fps:.1f}",
        (1920 - 300, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    cv2.imshow(window_name, panels.image_buffer)
    key = cv2.waitKey(10)
    if key == ord("q"):
        stop_control_thread = True
        control_thread.join()
        cv2.destroyAllWindows()
        for zed in zeds.values():
            zed.close()
        break

    end_time = time.time()
    loop_time = end_time - start_time
    loop_time_queue.append(loop_time)
    fps = 1.0 / np.mean(list(loop_time_queue))

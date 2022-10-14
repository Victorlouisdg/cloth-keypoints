from camera_toolkit.zed2i import Zed2i
import cv2
import pyzed.sl as sl
import numpy as np
from cloth_manipulation.manual_keypoints import ClothTransform
import torch
import threading
import time
import cloth_manipulation.camera_mapping as cm
from cloth_manipulation.gui import FourPanels
from cloth_manipulation.observers import KeypointObserver
from collections import deque

keypoint_observer =  KeypointObserver()

resolution = sl.RESOLUTION.HD720
control_image_crop_size = 600

low_resolution = False
if low_resolution:
    resolution = sl.RESOLUTION.VGA
    control_image_crop_size = 300

serial_numbers = [cm.CameraMapping.serial_top, cm.CameraMapping.serial_side, cm.CameraMapping.serial_front]
zeds = {n: Zed2i(resolution=resolution,serial_number=n, fps=30) for n in serial_numbers}

# Configure custom project-wide ClothTransform based on camera, resolution, etc.
_, h ,w = zeds[cm.CameraMapping.serial_top].get_rgb_image().shape
ClothTransform.crop_start_u = (w - control_image_crop_size) // 2
ClothTransform.crop_width = control_image_crop_size
ClothTransform.crop_start_v = (h - control_image_crop_size) // 2
ClothTransform.crop_height = control_image_crop_size

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

        keypoints = keypoint_observer.observe(control_image)
        control_visualization_image = keypoint_observer.visualize_last_observation()
        control_visualization_panel.fill_image_buffer(control_visualization_image)


control_thread = threading.Thread(target=control_loop)
control_thread.start()

print("Press q to quit.")

window_name = "GUI"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

loop_time_queue = deque(maxlen=15)
fps = -1

while True:
    start_time = time.time()
    images = {serial_number: zed.get_rgb_image() for serial_number, zed in zeds.items()}

    control_image = images[cm.CameraMapping.serial_top].copy()
    control_image_index += 1

    for serial_number, image in images.items():
        images[serial_number] =  Zed2i.image_shape_torch_to_opencv(image)

    panels.top_right.fill_image_buffer(images[cm.CameraMapping.serial_top])
    panels.bottom_left.fill_image_buffer(images[cm.CameraMapping.serial_front])
    panels.bottom_right.fill_image_buffer(images[cm.CameraMapping.serial_side])

    cv2.putText(panels.image_buffer, f"camera fps: {fps:.1f}", (1920 - 300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

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

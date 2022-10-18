import datetime
import os
import threading
import time
from collections import deque
from enum import IntEnum
from pathlib import Path

import cv2
import numpy as np
import pyzed.sl as sl
from camera_toolkit.reproject import reproject_to_world_z_plane
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.calibration import load_saved_calibration
from cloth_manipulation.camera_mapping import CameraMapping
from cloth_manipulation.controllers import ReorientAndFoldTowelController
from cloth_manipulation.gui import FourPanels, Panel
from cloth_manipulation.hardware.setup_hardware import setup_victor_louise
from cloth_manipulation.input_transform import InputTransform
from cloth_manipulation.observers import KeypointObserver

keypoint_observer = KeypointObserver()

victor_louise = setup_victor_louise()

world_to_camera = load_saved_calibration()

# resolution = sl.RESOLUTION.HD720

output_dir = Path(__file__).parent / "results" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(output_dir)
# output_csv = output_dir / "grasp_points.csv"
# csv_header = ["trial", ]


top_camera_resolution = sl.RESOLUTION.HD720
crop_size = 600
crop_vertical_offset = 40

if top_camera_resolution == sl.RESOLUTION.HD2K:
    crop_size = 1242
    crop_vertical_offset = 0

serial_numbers = [CameraMapping.serial_top, CameraMapping.serial_side, CameraMapping.serial_front]
resolutions = [top_camera_resolution, sl.RESOLUTION.HD720, sl.RESOLUTION.HD720]
zeds = {n: Zed2i(resolution=r, serial_number=n, fps=30) for r, n in zip(resolutions, serial_numbers)}

camera_matrix = zeds[CameraMapping.serial_top].get_camera_matrix()


# Configure custom project-wide InputTransform based on camera, resolution, etc.
init_image = zeds[CameraMapping.serial_top].get_rgb_image()
_, h, w = init_image.shape
InputTransform.crop_start_u = (w - crop_size) // 2
InputTransform.crop_width = crop_size
InputTransform.crop_start_v = (h - crop_size) // 2 + crop_vertical_offset
InputTransform.crop_height = crop_size

panels = FourPanels()

# Global vars for use in control thread
stop_control_thread = False
control_image_index = -1
control_image = None
top_left_panel = panels.top_left
trial = 0
max_trials = 6
controller = None


class Modes(IntEnum):
    CAMERA_FEED = 0
    PREVIEW_PLAN = 1
    FOLDING = 2


mode = 0  # Modes.CAMERA_FEED
prev_mode = None

already_detected = False


def control_loop(keypoint_observer):
    global stop_control_thread
    global control_image_index
    global control_image
    global top_left_panel
    # global keypoint_observer
    global mode
    global trial
    global max_trials
    global init_image
    global controller
    global world_to_camera
    global camera_matrix
    global prev_mode
    global victor_louise

    while not stop_control_thread:
        if init_image is not None:
            start = time.time()
            keypoint_observer.observe(init_image)
            print(f"Init observation took {time.time()-start:.2f}")
            init_image = None

        if control_image is None:
            time.sleep(0.1)
            print("No control image.")
            continue

        if controller is None or controller.finished:
            mode = Modes.CAMERA_FEED
            controller = ReorientAndFoldTowelController(victor_louise)

        _mode = mode  # copy mode locally so it cant change within a loop iteration

        keypoints = keypoint_observer.observe(control_image)
        keypoints_in_camera = InputTransform.reverse_transform_keypoints(np.array(keypoints))
        keypoints_in_world = reproject_to_world_z_plane(keypoints_in_camera, camera_matrix, world_to_camera)

        image = Zed2i.image_shape_torch_to_opencv(control_image)
        image = image.copy()
        if _mode == Modes.PREVIEW_PLAN or _mode == Modes.FOLDING:
            image = controller.visualize_plan(image, keypoints_in_world, world_to_camera, camera_matrix)

        if _mode == Modes.FOLDING and not prev_mode == Modes.FOLDING:
            trial += 1

        buffer = np.zeros_like(top_left_panel.image_buffer)
        Panel.fit_image_into_buffer(image, buffer)
        text = f"{Modes(_mode).name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        buffer = cv2.putText(buffer, text, (20, 40), font, 0.6, (0, 255, 0), 1)

        text = f"Trial: {trial}/{max_trials}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        buffer = cv2.putText(buffer, text, (20, 80), font, 1, (0, 255, 0), 2)

        top_left_panel.image_buffer[:, :, :] = buffer[:, :, :]

        if _mode == Modes.FOLDING:
            controller.act(keypoints_in_world)

        prev_mode = _mode


control_thread = threading.Thread(target=control_loop, args=(keypoint_observer,))
control_thread.start()

print("Press q to quit.")

window_name = "GUI"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


# def mouse_callback(event, x, y, flags, parm):
#     global mode
#     if event == cv2.EVENT_LBUTTONDOWN:
#         mode = mode + 1
#         mode %= len(Modes)

# cv2.setMouseCallback(window_name, mouse_callback)

loop_time_queue = deque(maxlen=15)
fps = -1


text = "Initializing"
font = cv2.FONT_HERSHEY_SIMPLEX
panels.top_left.image_buffer = cv2.putText(panels.top_left.image_buffer, text, (320, 300), font, 2, (0, 255, 0), 2)


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
    if key == ord("f") and mode == Modes.CAMERA_FEED:
        mode = Modes.FOLDING
    if key == ord("p"):
        mode = Modes.PREVIEW_PLAN
    if key == ord("c"):
        mode = Modes.CAMERA_FEED
    if key == ord("r"):
        mode = Modes.CAMERA_FEED
        controller = None
        control_thread = threading.Thread(target=control_loop, args=(keypoint_observer,))
        control_thread.start()

    end_time = time.time()
    loop_time = end_time - start_time
    loop_time_queue.append(loop_time)
    fps = 1.0 / np.mean(list(loop_time_queue))

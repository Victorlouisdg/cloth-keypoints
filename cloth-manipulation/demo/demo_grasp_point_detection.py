import csv
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
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.camera_mapping import CameraMapping
from cloth_manipulation.gui import FourPanels, Panel
from cloth_manipulation.input_transform import InputTransform
from cloth_manipulation.observers import KeypointObserver

keypoint_observer = KeypointObserver()

# resolution = sl.RESOLUTION.HD720

output_dir = (
    Path(__file__).parent / "results_grasp_point_detection" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)
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
max_trials = 10


class Modes(IntEnum):
    CAMERA_FEED = 0
    SINGLE_DETECTION = 1
    LIVE_DETECTION = 2


mode = 0  # Modes.CAMERA_FEED
already_detected = False


# noqa: C901
def control_loop(keypoint_observer):  # noqa: C901
    global stop_control_thread
    global control_image_index
    global control_image
    global top_left_panel
    # global keypoint_observer
    global mode
    global trial
    global max_trials
    global init_image

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

        _mode = mode  # copy mode locally so it cant change within a loop iteration

        if _mode != Modes.SINGLE_DETECTION:
            already_detected = False

        if _mode == Modes.CAMERA_FEED:
            image = InputTransform.transform_image(control_image)
            image = Zed2i.image_shape_torch_to_opencv(image)
            image = image.copy()
        elif _mode == Modes.SINGLE_DETECTION:
            if not already_detected:
                start = time.time()
                keypoint_observer.observe(control_image)
                print(f"Observation took {time.time()-start:.2f}")
                already_detected = True
                trial += 1
                cv2.imwrite(str(output_dir / f"trial_input_{trial}.png"), image)
                image = keypoint_observer.visualize_last_observation(competition_format=True)
                cv2.imwrite(str(output_dir / f"trial_annotated_{trial}.png"), image)

                csv_path = str(output_dir / f"trial_grasp_points_{trial}.csv")
                with open(csv_path, "w") as file:
                    writer = csv.writer(file)
                    for keypoint, approach_point in zip(
                        keypoint_observer.keypoints, keypoint_observer.approach_points
                    ):
                        row = [keypoint[0], keypoint[1], approach_point[0], approach_point[1]]
                        row = [int(coord) for coord in row]
                        writer.writerow(row)

            else:
                image = keypoint_observer.visualize_last_observation(competition_format=True)
        else:
            keypoint_observer.observe(control_image)
            image = keypoint_observer.visualize_last_observation()

        buffer = np.zeros_like(top_left_panel.image_buffer)
        Panel.fit_image_into_buffer(image, buffer)
        text = f"{Modes(_mode).name}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        buffer = cv2.putText(buffer, text, (20, 40), font, 0.6, (0, 255, 0), 1)

        text = f"Trial: {trial}/{max_trials}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        buffer = cv2.putText(buffer, text, (20, 80), font, 1, (0, 255, 0), 2)

        top_left_panel.image_buffer[:, :, :] = buffer[:, :, :]


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
    if key == ord("l"):
        mode = Modes.LIVE_DETECTION
    if key == ord("c"):
        mode = Modes.CAMERA_FEED
    if key == ord("s"):
        if mode == Modes.CAMERA_FEED:
            mode = Modes.SINGLE_DETECTION

    end_time = time.time()
    loop_time = end_time - start_time
    loop_time_queue.append(loop_time)
    fps = 1.0 / np.mean(list(loop_time_queue))

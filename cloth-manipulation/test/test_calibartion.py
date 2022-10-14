from weakref import WeakValueDictionary
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.calibration import load_saved_calibration
import cv2
import pyzed.sl as sl
import numpy as np
import time
import cloth_manipulation.camera_mapping as cm
from cloth_manipulation.gui import Panel, draw_center_circle, draw_world_axes
from collections import deque
from camera_toolkit.reproject import project_world_to_image_plane

resolution = sl.RESOLUTION.HD720
control_image_crop_size = 600

zed = Zed2i(resolution=resolution, serial_number=cm.CameraMapping.serial_top, fps=30)

# Configure custom project-wide ClothTransform based on camera, resolution, etc.
_, h, w = zed.get_rgb_image().shape
panel = Panel(np.zeros((h, w, 3), dtype=np.uint8))

print("Press q to quit.")

window_name = "GUI"
cv2.namedWindow(window_name)

loop_time_queue = deque(maxlen=15)
fps = -1

while True:
    start_time = time.time()
    image = zed.get_rgb_image()
    image = zed.image_shape_torch_to_opencv(image)
    image = image.copy()
    image = draw_center_circle(image)

    world_to_camera = load_saved_calibration()
    camera_matrix = zed.get_camera_matrix()
    image = draw_world_axes(image, world_to_camera, camera_matrix)
    panel.fill_image_buffer(image)

    text = f"fps: {fps:.1f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (255, 255, 255)
    cv2.putText(panel.image_buffer, text, (w - 200, 50), font, 1, white, 2)

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

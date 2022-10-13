from camera_toolkit.zed2i import Zed2i
import cv2
import pyzed.sl as sl
import numpy as np
from cloth_manipulation.detector import get_wandb_model
from cloth_manipulation.manual_keypoints import ClothTransform
from keypoint_detection.utils.visualization import overlay_image_with_heatmap
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap
import torch
import threading
import time


serial_numbers = {
    "top": 38633712,
    "side": 35357320,
    "front": 31733653,
}
resolution = sl.RESOLUTION.HD720

zed_top = Zed2i(resolution=resolution, serial_number=serial_numbers["top"])
zed_side = Zed2i(resolution=resolution, serial_number=serial_numbers["side"])
zed_front = Zed2i(resolution=resolution, serial_number=serial_numbers["front"])

zeds = [zed_top, zed_side, zed_front]

shape_1080p = (1080, 1920)
shape_540p = (540, 960)
display_image = np.zeros((*shape_1080p, 3), dtype=np.uint8)

stop_inference = False
most_recent_top_down_image = None
most_recent_inference = None

checkpoint_reference = "airo-box-manipulation/iros2022/model-wy78a0el:v3"  # synthetic
keypoint_detector = get_wandb_model(checkpoint_reference)

h, w = 720, 1280
new_size = 600
ClothTransform.crop_start_u = (w - new_size) // 2
ClothTransform.crop_width = new_size
ClothTransform.crop_start_v = (h - new_size) // 2
ClothTransform.crop_height = new_size

def inference_loop():
    global stop_inference
    global most_recent_inference
    global most_recent_top_down_image
    print("Inference")

    while not stop_inference:
        image = most_recent_top_down_image
        if image is None:
            time.sleep(1)
            print("No image in thread.")
            continue
        transformed_image = ClothTransform.transform_image(image)
        image_batched = torch.Tensor(transformed_image).unsqueeze(0) / 255.0
        with torch.no_grad():
            heatmap = keypoint_detector(image_batched)

        heatmap_channel_batched = heatmap.squeeze(1)
        heatmap_channel = heatmap_channel_batched.squeeze(0)

        overlayed = overlay_image_with_heatmap(image_batched, heatmap_channel_batched)
        overlayed = overlayed.squeeze(0).numpy()
        overlayed = Zed2i.image_shape_torch_to_opencv(overlayed)
        overlayed = overlayed.copy()

        keypoints = get_keypoints_from_heatmap(
            heatmap_channel.cpu(), min_keypoint_pixel_distance=4, max_keypoints=4
        )
        for keypoint in keypoints:
            overlayed = cv2.circle(overlayed, keypoint, 6, (0, 1, 0))

        overlayed *= 255.0
        overlayed = overlayed.astype(np.uint8) 
        most_recent_inference = overlayed.copy()


inference_thread = threading.Thread(target=inference_loop)
inference_thread.start()


print("Press q to quit.")

window_name = "GUI"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while True:
    images = [zed.get_rgb_image() for zed in zeds]

    most_recent_top_down_image = images[0].copy()

    images = [Zed2i.image_shape_torch_to_opencv(image) for image in images]
    images = [cv2.resize(image, tuple(reversed(shape_540p))) for image in images]

    display_image[0:540, 960:, :] = images[0]
    display_image[540:, 0:960, :] = images[1]
    display_image[540:, 960:, :] = images[2]

    if most_recent_inference is not None:
        image = most_recent_inference
        image = cv2.resize(image, (540, 540))
        start_u = (960 - 540) // 2
        end_u = start_u+540
        display_image[:540, start_u:end_u] = image

    cv2.imshow(window_name, display_image)
    key = cv2.waitKey(10)
    if key == ord("q"):
        stop_inference = True
        inference_thread.join()
        cv2.destroyAllWindows()
        for zed in zeds:
            zed.close()
        break

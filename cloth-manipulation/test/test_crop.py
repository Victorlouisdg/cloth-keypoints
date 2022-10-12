"""A simple script that shows a ZED camera video feeds and draws the rectangle that will be cropped from it."""
from camera_toolkit.zed2i import Zed2i
import cv2
import pyzed.sl as sl
from cloth_manipulation.manual_keypoints import ClothTransform
import numpy as np

green_cv2 = (0, 255, 0)

serial_numbers = {
    "top": 38633712,
    "side": 35357320,
    "front": 31733653,
}


def draw_cloth_transform_rectangle(image_full_size) -> np.ndarray:
    u_top = ClothTransform.crop_start_u
    u_bottom = u_top + ClothTransform.crop_width
    v_top = ClothTransform.crop_start_v
    v_bottom = v_top + ClothTransform.crop_height

    top_left = (u_top, v_top)
    bottom_right = (u_bottom, v_bottom)

    image = cv2.rectangle(
        image_full_size, top_left, bottom_right, green_cv2, thickness=2
    )
    return image


def draw_center_circle(image) -> np.ndarray:
    h, w, _ = image.shape
    center_u = w // 2
    center_v = h // 2
    center = (center_u, center_v)
    radius = h // 100
    image = cv2.circle(image, center, radius, green_cv2, thickness=2)
    return image


resolution = sl.RESOLUTION.HD720

zed = Zed2i(resolution=resolution, serial_number=serial_numbers["top"])
image = zed.get_rgb_image("right")
image = zed.image_shape_torch_to_opencv(image)

print("image_shape", image.shape)
h, w, c = image.shape

new_size = 600
ClothTransform.crop_start_u = (w - new_size) // 2
ClothTransform.crop_width = new_size
ClothTransform.crop_start_v = (h - new_size) // 2
ClothTransform.crop_height = new_size

print("Press q to quit.")

while True:
    image = zed.get_rgb_image()
    image = zed.image_shape_torch_to_opencv(image)
    image = image.copy()
    image = draw_cloth_transform_rectangle(image)
    image = draw_center_circle(image)
    cv2.imshow("Image", image)
    key = cv2.waitKey(10)
    if key == ord("q"):
        cv2.destroyAllWindows()
        zed.close()
        break

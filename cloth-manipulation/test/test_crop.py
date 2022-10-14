"""A simple script that shows a ZED camera video feeds and draws the rectangle that will be cropped from it."""
from camera_toolkit.zed2i import Zed2i
import cv2
import pyzed.sl as sl
from cloth_manipulation.manual_keypoints import ClothTransform
import cloth_manipulation.camera_mapping as cm
from cloth_manipulation.gui import draw_cloth_transform_rectangle, draw_center_circle

resolution = sl.RESOLUTION.HD720

zed = Zed2i(resolution=resolution, serial_number=cm.CameraMapping.serial_top)
image = zed.get_rgb_image()
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

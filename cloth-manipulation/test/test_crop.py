"""A simple script that shows a ZED camera video feeds and draws the rectangle that will be cropped from it."""
import cv2
import pyzed.sl as sl
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.camera_mapping import CameraMapping
from cloth_manipulation.gui import draw_cloth_transform_rectangle
from cloth_manipulation.input_transform import InputTransform

resolution = sl.RESOLUTION.HD720

zed = Zed2i(resolution=resolution, serial_number=CameraMapping.serial_top)
image = zed.get_rgb_image()
image = zed.image_shape_torch_to_opencv(image)

print("image_shape", image.shape)
h, w, c = image.shape

new_size = 600
InputTransform.crop_start_u = (w - new_size) // 2
InputTransform.crop_width = new_size
InputTransform.crop_start_v = (h - new_size) // 2 + 40
InputTransform.crop_height = new_size

print("Press q to quit.")

while True:
    image = zed.get_rgb_image()
    transformed_image = InputTransform.transform_image(image)

    image = zed.image_shape_torch_to_opencv(image)
    transformed_image = zed.image_shape_torch_to_opencv(transformed_image)
    image = image.copy()
    transformed_image = transformed_image.copy()

    image = draw_cloth_transform_rectangle(image)

    # image = draw_center_circle(image)
    cv2.imshow("Image", image)
    cv2.imshow("Transformed Image", transformed_image)

    key = cv2.waitKey(10)
    if key == ord("q"):
        cv2.destroyAllWindows()
        zed.close()
        break
    if key == ord("s"):
        cv2.imwrite("image.png", transformed_image)

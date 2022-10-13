from camera_toolkit.zed2i import Zed2i
import cv2
import pyzed.sl as sl
import numpy as np
import cloth_manipulation.camera_mapping as cm

resolution = sl.RESOLUTION.HD720

zed_top = Zed2i(resolution=resolution, serial_number=cm.CameraMapping.serial_top)
zed_side = Zed2i(resolution=resolution, serial_number=cm.CameraMapping.serial_side)
zed_front = Zed2i(resolution=resolution, serial_number=cm.CameraMapping.serial_front)

zeds = [zed_top, zed_side, zed_front]

print("Press q to quit.")

window_name = "GUI"
cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)

while True:
    images = [zed.get_rgb_image() for zed in zeds]
    images = [zed.image_shape_torch_to_opencv(image) for image, zed in zip(images, zeds)]
    # image = np.vstack(images)
    # print(image.shape)

    shape_1080p = (1080, 1920)
    shape_540p = (540, 960)

    images = [cv2.resize(image, tuple(reversed(shape_540p))) for image in images]


    display_image = np.zeros((*shape_1080p, 3), dtype=np.uint8)
    display_image[0:540, 960:, :] = images[0]
    display_image[540:, 0:960, :] = images[1]
    display_image[540:, 960:, :] = images[2]


    cv2.imshow(window_name, display_image)
    key = cv2.waitKey(10)
    if key == ord("q"):
        cv2.destroyAllWindows()
        for zed in zeds:
            zed.close()
        break

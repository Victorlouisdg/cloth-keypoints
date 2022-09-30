import cv2
import numpy as np
from camera_toolkit.zed2i import Zed2i


def get_manual_keypoints(num_keypoints: int = 4):
    """function to capture image and select some keypoints manually, which allows to test the folding w/o the state estimation"""
    # opencv mouseclick registration
    clicked_coords = []

    def clicked_callback_cv(event, x, y, flags, param):
        global u_clicked, v_clicked
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print(f"clicked on {x}, {y}")
            clicked_coords.append(np.array([x, y]))

    # open ZED (and assert it is available)
    Zed2i.list_camera_serial_numbers()
    zed = Zed2i()

    # capture image
    img = zed.get_rgb_image()
    img = zed.image_shape_torch_to_opencv(img)

    # mark the keypoints in image plane by clicking
    cv2.imshow("img", img)
    cv2.setMouseCallback("img", clicked_callback_cv)

    while True:
        print(f"double click to select{num_keypoints} a keypoint; press any key after you are finished")

        cv2.waitKey(0)
        if len(clicked_coords) > num_keypoints:
            raise IndexError("too many keypoint clicked, aborting.")
        elif len(clicked_coords) == num_keypoints:
            break

    cv2.destroyAllWindows()

    zed.close()

    return clicked_coords


if __name__ == "__main__":
    print(get_manual_keypoints(4))

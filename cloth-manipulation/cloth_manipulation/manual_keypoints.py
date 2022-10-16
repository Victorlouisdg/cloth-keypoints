from typing import List, Tuple

import cv2
import numpy as np


def get_manual_keypoints(image: np.ndarray) -> List[Tuple[int, int]]:
    """Opens an OpenCV window and lets you interactively select a list of keypoints.
    Usage:
    * left-click: add a keypoint
    * backspace: remove the last keypoint
    * enter: confirm
    * q: quit and return no keypoints
    """
    window_name = "GUI"
    cv2.namedWindow(window_name)

    keypoints = []

    def mouse_callback(event, x, y, flags, parm):
        if event == cv2.EVENT_LBUTTONDOWN:
            keypoints.append((x, y))

    cv2.setMouseCallback(window_name, mouse_callback)
    while True:
        _image = image.copy()
        for i, clicked_point in enumerate(keypoints):
            _image = cv2.circle(_image, clicked_point, 5, (0, 255, 0), thickness=1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_position = np.array(clicked_point) + [5, -5]
            cv2.putText(_image, str(i), text_position, font, 1, (0, 255, 0), 2)

        cv2.imshow(window_name, _image)
        key = cv2.waitKey(10)
        if key == 8:  # backspace
            keypoints.pop()
        if key == 13:  # enter
            return keypoints
        if key == ord("q"):
            cv2.destroyWindow(window_name)
            return []

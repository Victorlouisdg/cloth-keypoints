import cv2
import numpy as np
from camera_toolkit.zed2i import Zed2i

from pathlib import Path
import pickle



# load camera to marker transform
with open(Path(__file__).parent / "marker.pickle", "rb") as f:
    aruco_in_camera_position, aruco_in_camera_orientation = pickle.load(f)
print(f"{aruco_in_camera_position=}")
print(f"{aruco_in_camera_orientation=}")

# get camera extrinsics transform
aruco_in_camera_transform = np.eye(4)
aruco_in_camera_transform[:3, :3] = aruco_in_camera_orientation
aruco_in_camera_transform[:3, 3] = aruco_in_camera_position



def get_manual_keypoints(zed: Zed2i, num_keypoints: int = 4):
    """function to capture image and select some keypoints manually, which allows to test the folding w/o the state estimation"""
    # opencv mouseclick registration
    clicked_coords = []

    def clicked_callback_cv(event, x, y, flags, param):
        global u_clicked, v_clicked
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print(f"clicked on {x}, {y}")
            clicked_coords.append(np.array([x, y]))


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


    return clicked_coords


if __name__ == "__main__":
    from camera_toolkit.reproject import reproject_to_world_z_plane


    # open ZED (and assert it is available)
    Zed2i.list_camera_serial_numbers()
    zed = Zed2i()
    keypoints_in_camera = np.array(get_manual_keypoints(zed, 4))
    keypoints_in_world = reproject_to_world_z_plane(keypoints_in_camera,zed.get_camera_matrix(),aruco_in_camera_transform)
    print(keypoints_in_world)

import numpy as np
from camera_toolkit.reproject import reproject_to_world_z_plane
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.manual_keypoints import ClothTransform
from cloth_manipulation.motion_primitives.pull import execute_pull_primitive, TowelReorientPull
from cloth_manipulation.setup_hw import setup_hw
import torch
from cloth_manipulation.towel import reorient_towel

if __name__ == "__main__":
    dual_arm = setup_hw()
    from camera_toolkit.reproject import reproject_to_world_z_plane
    from cloth_manipulation.manual_keypoints import aruco_in_camera_transform, get_manual_keypoints

    # open ZED (and assert it is available)
    Zed2i.list_camera_serial_numbers()
    zed = Zed2i()
    reorient_towel(zed, dual_arm)
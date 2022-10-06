import numpy as np
from camera_toolkit.reproject import reproject_to_world_z_plane
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.manual_keypoints import ClothTransform
from cloth_manipulation.motion_primitives.pull import execute_pull_primitive, TowelReorientPull
from cloth_manipulation.setup_hw import setup_hw
import torch
from cloth_manipulation.ur_robotiq_dual_arm_interface import DualArmUR
from cloth_manipulation.manual_keypoints import get_manual_keypoints, aruco_in_camera_transform

def reorient_towel(zed: Zed2i, dual_arm: DualArmUR , keypoint_detector: torch.nn.Module = None):
        # L move to home to avoid collisions with other robot?
        dual_arm.dual_moveL(
            dual_arm.victor_ur.home_pose, dual_arm.louise_ur.home_pose, vel=2 * dual_arm.DEFAULT_LINEAR_VEL
        )
        while True:
            dual_arm.dual_moveL(
                dual_arm.victor_ur.out_of_way_pose, dual_arm.louise_ur.out_of_way_pose, vel=2 * dual_arm.DEFAULT_LINEAR_VEL
            )
            img = zed.get_rgb_image()
            transformed_image = ClothTransform.transform_image(img)
            if not keypoint_detector:
                transformed_cv_img = Zed2i.image_shape_torch_to_opencv(transformed_image)
                transformed_keypoints = np.array(get_manual_keypoints(transformed_cv_img, 4))
            else:
                transformed_keypoints = keypoint_detector()

            keypoints_in_camera = ClothTransform.reverse_transform_keypoints(transformed_keypoints)
            keypoints_in_world = reproject_to_world_z_plane(
                keypoints_in_camera, zed.get_camera_matrix(), aruco_in_camera_transform
            )

            pullprimitive = TowelReorientPull(keypoints_in_world)
            if pullprimitive.average_corner_error() < 0.05:
                print("pull was less than 5cm, no need to execute")
                break
            print(pullprimitive)
            execute_pull_primitive(pullprimitive, dual_arm)

        dual_arm.dual_moveL(
            dual_arm.victor_ur.home_pose, dual_arm.louise_ur.home_pose, vel=2 * dual_arm.DEFAULT_LINEAR_VEL
        )


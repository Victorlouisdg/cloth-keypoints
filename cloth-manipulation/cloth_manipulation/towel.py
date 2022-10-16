import numpy as np
import torch
from camera_toolkit.reproject import reproject_to_world_z_plane
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.manual_keypoints import ClothTransform, aruco_in_camera_transform, get_manual_keypoints
from cloth_manipulation.motion_primitives.fold_execution import execute_dual_fold_trajectories
from cloth_manipulation.motion_primitives.fold_trajectory_parameterization import CircularFoldTrajectory
from cloth_manipulation.motion_primitives.pull import TowelReorientPull, execute_pull_primitive
from cloth_manipulation.hardware.ur_robotiq_dual_arm_interface import DualArmUR
from cloth_manipulation.geometry import get_ordered_keypoints

def get_towel_keypoints(zed: Zed2i, keypoint_detector: torch.nn.Module = None):
    img = zed.get_rgb_image()
    transformed_image = ClothTransform.transform_image(img)
    if not keypoint_detector:
        transformed_cv_img = Zed2i.image_shape_torch_to_opencv(transformed_image)
        transformed_keypoints = np.array(get_manual_keypoints(transformed_cv_img, 4))
    else:
        transformed_keypoints = keypoint_detector(img.unsqueeze(0)).squeeze(0)

    keypoints_in_camera = ClothTransform.reverse_transform_keypoints(transformed_keypoints)
    keypoints_in_world = reproject_to_world_z_plane(
        keypoints_in_camera, zed.get_camera_matrix(), aruco_in_camera_transform
    )
    return keypoints_in_world


def reorient_towel(zed: Zed2i, dual_arm: DualArmUR, keypoint_detector: torch.nn.Module = None):
    dual_arm.dual_moveL(
        dual_arm.victor_ur.home_pose, dual_arm.louise_ur.home_pose, vel=2 * dual_arm.DEFAULT_LINEAR_VEL
    )
    while True:
        dual_arm.dual_moveL(
            dual_arm.victor_ur.out_of_way_pose, dual_arm.louise_ur.out_of_way_pose, vel=2 * dual_arm.DEFAULT_LINEAR_VEL
        )
        keypoints_in_world = get_towel_keypoints(zed,keypoint_detector)
        pullprimitive = TowelReorientPull(keypoints_in_world, dual_arm)
        if pullprimitive.average_corner_error() < 0.05:
            print("pull was less than 5cm, no need to execute")
            break
        print(pullprimitive)
        execute_pull_primitive(pullprimitive, dual_arm)

    dual_arm.dual_moveL(
        dual_arm.victor_ur.home_pose, dual_arm.louise_ur.home_pose, vel=2 * dual_arm.DEFAULT_LINEAR_VEL
    )

def inset_short_edge(left:np.array,right:np.array, margin =  0.04):
    """
    Inset the folding keypoints over the short distance (so end-end or start-start)
    for more robust folding.
    """
    direction = left-right
    direction /= np.linalg.norm(direction)

    left -= margin * direction
    right += margin  * direction

    return left, right
    

def fold_towel_once(zed: Zed2i, dual_arm: DualArmUR, keypoint_detector: torch.nn.Module = None):
    

    # assuming the orientation is already OK
    keypoints_in_world = get_towel_keypoints(zed,keypoint_detector)
    corners = np.array(keypoints_in_world)
    corners = get_ordered_keypoints(corners)
    end_louise, end_victor, start_victor, start_louise = corners

    end_louise, end_victor = inset_short_edge(end_louise, end_victor)
    start_louise, start_victor = inset_short_edge(start_louise, start_victor)

    fold_trajectory_victor = CircularFoldTrajectory(start_victor, end_victor)
    fold_trajectory_louise = CircularFoldTrajectory(start_louise, end_louise)

    execute_dual_fold_trajectories(fold_trajectory_victor, fold_trajectory_louise, dual_arm)

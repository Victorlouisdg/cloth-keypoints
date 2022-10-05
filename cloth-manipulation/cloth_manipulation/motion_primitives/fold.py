import logging
from typing import List

import numpy as np
from cloth_manipulation.motion_primitives.fold_trajectory_parameterization import FoldTrajectory
from cloth_manipulation.ur_robotiq_dual_arm_interface import UR, DualArmUR, homogeneous_pose_to_position_and_rotvec

logger = logging.Logger(__file__)


def make_robot_formatted_trajectory_from_path(path: List[np.ndarray]):
    assert path[0].shape == (4, 4)  # homogeneous coords
    trajectory_in_robot_format = []
    for pose in path:
        waypoint = np.array(
            list(homogeneous_pose_to_position_and_rotvec(pose))
            + [UR.DEFAULT_LINEAR_VEL, UR.DEFAULT_LINEAR_ACC, UR.DEFAULT_BLEND]
        )
        trajectory_in_robot_format.append(waypoint)
    return trajectory_in_robot_format


def execute_single_fold_line(fold_line: FoldTrajectory, robot: UR):

    robot.moveL(robot.home_pose, vel=2 * robot.DEFAULT_LINEAR_VEL)
    # move to pregrasp
    pregrasp_pose = homogeneous_pose_to_position_and_rotvec(fold_line.get_pregrasp_pose())
    logger.debug(f"{pregrasp_pose=}")
    print(pregrasp_pose)
    robot.moveL(pregrasp_pose)
    # graps
    robot.moveL(homogeneous_pose_to_position_and_rotvec(fold_line.get_grasp_pose()))
    robot.gripper.close()

    # execute fold
    path = fold_line.get_fold_path()
    trajectory = make_robot_formatted_trajectory_from_path(path)
    robot.moveP(trajectory)
    # release
    robot.moveL(homogeneous_pose_to_position_and_rotvec(fold_line.get_fold_retreat_pose()))
    robot.gripper.open()
    robot.moveL(robot.home_pose, vel=2 * robot.DEFAULT_LINEAR_VEL)


def execute_dual_fold_lines(fold_line_victor: FoldTrajectory, fold_line_louise: FoldTrajectory, dual_arm: DualArmUR):

    # move to home pose
    dual_arm.dual_moveL(
        dual_arm.victor_ur.home_pose, dual_arm.louise_ur.home_pose, vel=2 * dual_arm.DEFAULT_LINEAR_VEL
    )

    # move to pregrasp
    dual_arm.dual_moveL(
        homogeneous_pose_to_position_and_rotvec(fold_line_victor.get_pregrasp_pose()),
        homogeneous_pose_to_position_and_rotvec(fold_line_louise.get_pregrasp_pose()),
    )

    # grasp
    dual_arm.dual_moveL(
        homogeneous_pose_to_position_and_rotvec(fold_line_victor.get_grasp_pose()),
        homogeneous_pose_to_position_and_rotvec(fold_line_louise.get_grasp_pose()),
    )
    dual_arm.victor_ur.gripper.close()
    dual_arm.louise_ur.gripper.close()

    # execute fold
    path_victor = fold_line_victor.get_fold_path()
    trajectory_victor = make_robot_formatted_trajectory_from_path(path_victor)
    path_louise = fold_line_louise.get_fold_path()
    trajectory_louise = make_robot_formatted_trajectory_from_path(path_louise)
    dual_arm.dual_moveP(trajectory_victor, trajectory_louise)

    # release
    dual_arm.dual_moveL(
        homogeneous_pose_to_position_and_rotvec(fold_line_victor.get_fold_retreat_pose()),
        homogeneous_pose_to_position_and_rotvec(fold_line_louise.get_fold_retreat_pose()),
    )
    dual_arm.victor_ur.gripper.open()
    dual_arm.louise_ur.gripper.open()

    # move to home pose
    dual_arm.dual_moveL(dual_arm.victor_ur.home_pose, dual_arm.louise_ur.home_pose, vel=2 * UR.DEFAULT_LINEAR_VEL)

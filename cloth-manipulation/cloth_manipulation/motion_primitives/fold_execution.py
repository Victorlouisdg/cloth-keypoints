from cloth_manipulation.hardware.base_classes import DualArm, RobotArm
from cloth_manipulation.motion_primitives.fold_trajectory_parameterization import FoldTrajectory


def execute_single_fold_trajectory(fold: FoldTrajectory, robot: RobotArm):
    # go to home pose
    robot.move_tcp(robot.home_pose)
    # move to pregrasp
    pregrasp_pose = fold.get_pregrasp_pose()
    robot.move_tcp(pregrasp_pose)
    # grasp
    robot.move_tcp_linear(fold.get_grasp_pose(), robot.LINEAR_SPEED, robot.LINEAR_ACCELERATION)
    robot.gripper.close()

    # execute fold
    path = fold.get_fold_path()
    robot.move_tcp_linear_path(path, robot.LINEAR_SPEED, robot.LINEAR_ACCELERATION)

    # release
    robot.gripper.open()
    robot.move_tcp_linear(fold.get_fold_retreat_pose(), robot.LINEAR_SPEED, robot.LINEAR_ACCELERATION)

    robot.move_tcp(robot.home_pose)


def execute_dual_fold_trajectories(fold_left: FoldTrajectory, fold_right: FoldTrajectory, dual_arm: DualArm):
    # move to home poses
    dual_arm.dual_move_tcp(dual_arm.left.home_pose, dual_arm.right.home_pose)

    dual_arm.dual_gripper_open()

    # move freely to pregrasp
    dual_arm.dual_move_tcp(
        fold_left.get_pregrasp_pose(),
        fold_right.get_pregrasp_pose(),
    )

    # grasp with linear motion
    dual_arm.dual_move_tcp_linear(
        fold_left.get_grasp_pose(),
        fold_right.get_grasp_pose(),
        dual_arm.left.LINEAR_SPEED,
        dual_arm.left.LINEAR_ACCELERATION,
    )

    dual_arm.dual_gripper_close()

    # execute fold paths
    path_left = fold_left.get_fold_path()
    path_right = fold_right.get_fold_path()
    dual_arm.dual_move_tcp_linear_path(
        path_left,
        path_right,
        dual_arm.left.LINEAR_SPEED,
        dual_arm.left.LINEAR_ACCELERATION,
    )

    dual_arm.dual_gripper_move_to_position(0.6)

    # release and retreat with a linear motion
    dual_arm.dual_move_tcp_linear(
        fold_left.get_fold_retreat_pose(),
        fold_right.get_fold_retreat_pose(),
        dual_arm.left.LINEAR_SPEED,
        dual_arm.left.LINEAR_ACCELERATION,
    )

    # move to home pose
    dual_arm.dual_move_tcp(dual_arm.left.home_pose, dual_arm.right.home_pose)

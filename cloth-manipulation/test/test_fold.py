import logging
import time

import numpy as np
from cloth_manipulation.motion_primitives.fold import execute_dual_fold_lines
from fold_line_parameterization import CircularFoldLine
from scipy.spatial.transform import Rotation as R
from ur_robotiq_dual_arm_interface import UR, DualArmUR, RobotiqTCP, RobotiqUSB

################
# no common world frame! all coords in respective robot frames for testing.
#################
logging.basicConfig(level=logging.DEBUG)

ip_victor = "10.42.0.162"
ip_louise = "10.42.0.163"

gripper_victor = RobotiqTCP(ip_victor)
gripper_louise = RobotiqUSB()
victor = UR(ip_victor, gripper_victor)
louise = UR(ip_louise, gripper_louise)
dual_arm = DualArmUR(victor, louise)
home_orientation = list(R.from_euler("yz", [np.pi, -np.pi / 2]).as_rotvec())
home_position_victor = [0.2, -0.1, 0.2]
home_position_louise = [-0.2, -0.1, 0.2]

home_victor_waypoint = np.array(home_position_victor + home_orientation)
home_waypoint_louise = np.array(home_position_louise + home_orientation)

time.sleep(4)

dual_arm.dual_moveJ_IK(pose_in_world_victor=home_victor_waypoint, pose_in_world_louise=home_waypoint_louise)

fold_line_victor = CircularFoldLine(np.array([0.25, -0.2, 0.05]), np.array([0.25, 0.2, 0.05]))
fold_line_louise = CircularFoldLine(np.array([-0.25, -0.2, 0.05]), np.array([-0.2, 0.2, 0.05]))

# execute_single_fold_line(fold_line_victor,victor)
execute_dual_fold_lines(fold_line_victor, fold_line_louise, dual_arm)

dual_arm.dual_moveJ_IK(pose_in_world_victor=home_victor_waypoint, pose_in_world_louise=home_waypoint_louise)

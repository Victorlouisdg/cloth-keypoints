import numpy as np
from scipy.spatial.transform import Rotation as R
from cloth_manipulation.ur_robotiq_dual_arm_interface import UR, DualArmUR, RobotiqTCP

ip_victor = "10.42.0.162"
ip_louise = "10.42.0.163"

gripper_victor = RobotiqTCP(ip_victor)
gripper_louise = None

victor = UR(ip_victor, gripper_victor)
louise = UR(ip_louise, gripper_louise)

dual_arm = DualArmUR(victor, louise)


home_orientation = list(R.from_euler("yz", [np.pi, -np.pi / 2]).as_rotvec())
home_position_victor = [0.2, -0.1, 0.2]
home_position_louise = [-0.2, -0.1, 0.2]

home_victor_waypoint = np.array(home_position_victor + home_orientation)
home_waypoint_louise = np.array(home_position_louise + home_orientation)

dual_arm.dual_moveJ_IK(pose_in_world_victor=home_victor_waypoint, pose_in_world_louise=home_waypoint_louise)


pre_fold_waypoint_victor = np.array([0.15, -0.25, 0.15] + home_orientation)
pre_fold_waypoint_louise = np.array([-0.15, -0.25, 0.15] + home_orientation)
dual_arm.dual_moveL(pre_fold_waypoint_victor, pre_fold_waypoint_louise, vel=0.14)

pre_fold_waypoint_victor = np.array([0.15, 0.25, 0.15] + home_orientation)
pre_fold_waypoint_louise = np.array([-0.15, 0.25, 0.15] + home_orientation)
dual_arm.dual_moveJ_IK(pre_fold_waypoint_victor, pre_fold_waypoint_louise, vel=0.14)


victor.moveL(home_victor_waypoint)
louise.moveJ_IK(home_waypoint_louise)

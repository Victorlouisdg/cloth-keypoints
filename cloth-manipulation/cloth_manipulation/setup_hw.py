"""
script for initializing the HW setup.

Both robots are on the X-axis, with manually measured offsets of + (victor) and -(louise) 39cm.
"""

from cloth_manipulation.ur_robotiq_dual_arm_interface import RobotiqTCP, DualArmUR, UR
from scipy.spatial.transform import Rotation as R
import numpy as np


def setup_hw() -> DualArmUR:

    ip_victor = "10.42.0.162"
    ip_louise = "10.42.0.163"

    home_orientation = list(R.from_euler("yz", [np.pi, -np.pi / 2]).as_rotvec())

    home_position_victor = [0.2-0.39, -0.1, 0.2]
    home_position_louise = [-0.2+0.39, -0.1, 0.2]
    out_of_way_position_victor = [-0.05-0.39, -0.2, 0.2]
    out_of_way_position_louise = [0.05+0.39, -0.2, 0.2]


    home_pose_victor = np.array(home_position_victor + home_orientation)
    home_pose_louise = np.array(home_position_louise + home_orientation)
    out_of_way_pose_victor = np.array(out_of_way_position_victor + home_orientation)
    out_of_way_pose_louise = np.array(out_of_way_position_louise + home_orientation)


    gripper_victor = RobotiqTCP(ip_victor)
    gripper_louise = RobotiqTCP(ip_louise)

    victor = UR(ip_victor, gripper_victor,[-0.39,0.0,+0.006])
    louise = UR(ip_louise, gripper_louise,[0.39,0.0,+0.006])

    victor.out_of_way_pose = out_of_way_pose_victor
    victor.home_pose = home_pose_victor
    louise.out_of_way_pose = out_of_way_pose_louise
    louise.home_pose = home_pose_louise

    dual_arm = DualArmUR(victor, louise)
    return dual_arm

if __name__ == "__main__":
    dual_arm = setup_hw()
    dual_arm.victor_ur.gripper.open()
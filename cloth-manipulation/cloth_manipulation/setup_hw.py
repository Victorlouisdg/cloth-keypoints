from cloth_manipulation.ur_robotiq_dual_arm_interface import RobotiqTCP, DualArmUR, UR
import numpy as np 
from scipy.spatial.transform import Rotation as R

home_orientation = list(R.from_euler("yz", [np.pi, -np.pi / 2]).as_rotvec())
home_position_victor = [0.2-0.39, -0.1, 0.2]
home_position_louise = [-0.2+0.39, -0.1, 0.2]

home_victor_waypoint = np.array(home_position_victor + home_orientation)
home_waypoint_louise = np.array(home_position_louise + home_orientation)

def setup_hw() -> DualArmUR:
    ip_victor = "10.42.0.162"
    ip_louise = "10.42.0.163"

    gripper_victor = RobotiqTCP(ip_victor)
    gripper_louise = RobotiqTCP(ip_louise)

    victor = UR(ip_victor, gripper_victor,[-0.39,0.0,+0.006])
    louise = UR(ip_louise, gripper_louise,[0.39,0.0,+0.006])

    dual_arm = DualArmUR(victor, louise)
    return dual_arm

if __name__ == "__main__":
    dual_arm = setup_hw()
    dual_arm.victor_ur.gripper.open()
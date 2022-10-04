from cloth_manipulation.ur_robotiq_dual_arm_interface import RobotiqTCP, DualArmUR, UR

ip_victor = "10.42.0.162"
ip_louise = "10.42.0.163"

gripper_victor = RobotiqTCP(ip_victor)
gripper_louise = RobotiqTCP(ip_louise)

victor = UR(ip_victor, gripper_victor)
louise = UR(ip_louise, gripper_louise)

dual_arm = DualArmUR(victor, louise)
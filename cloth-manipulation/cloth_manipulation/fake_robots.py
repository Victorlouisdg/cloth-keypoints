from cloth_manipulation.ur_robotiq_dual_arm_interface import Gripper


class FakeGripper(Gripper):
    def open(self):
        print("Opening Gripper")

    def close(self):
        print("Opening Gripper")


class FakeRobot:
    def __init__(self, robot_in_world_position=[0, 0, 0], gripper=FakeGripper()):
        self.gripper = gripper
        self.robot_in_world_position = robot_in_world_position


class FakeDualArm:
    def __init__(self, victor, louise) -> None:
        self.victor_ur = victor
        self.louise_ur = louise

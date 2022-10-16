from cloth_manipulation.hardware.base_classes import Gripper
from robotiq2f import Robotiq2F85TCP


# This class is only a thin wrapper arround our own Robotiq2F85TCP.
# Long term, that class would ideally inherit from Gripper so this class becomes unnecessary.
class Robotiq2F85(Gripper):
    def __init__(self, robot_ip) -> None:
        super().__init__()
        self.gripper = Robotiq2F85TCP(robot_ip)
        self.gripper.activate_gripper()

    def open(self):
        self.gripper.open()

    def close(self):
        self.gripper.close()

    def move_to_position(self, position):
        self.gripper.move_to_position(int(position * 255))

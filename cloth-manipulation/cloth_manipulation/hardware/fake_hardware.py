from cloth_manipulation.hardware.base_classes import Gripper, RobotArm
from typing import List, Optional
import numpy as np


class FakeGripper(Gripper):
    def open(self):
        print("Opening gripper.")

    def close(self):
        print("Closing gripper.")

    def move_to_position(self, position):
        print(f"Moving gripper to position {position}.")


class FakeArm(RobotArm):
    # Default settings copied from the UR. Allows testing the pull primitive. In the future I would like these removed.
    LINEAR_SPEED = 0.1  # m/s
    LINEAR_ACCELERATION = 0.4  # m/s^2
    JOINT_SPEED = 0.4  # rad/s
    JOINT_ACCELERATION = 0.8  # rad/s^2
    BLEND_RADIUS = 0.01  # m?
    MIN_SAFE_TCP_TO_BASE_DISTANCE = 0.08  # m

    def __init__(
        self,
        name: str,
        robot_in_world_pose: np.ndarray,
        home_pose: np.ndarray,
        out_of_way_pose: np.ndarray,
        gripper: Optional[Gripper] = None,
    ):
        super().__init__(name, robot_in_world_pose, home_pose, out_of_way_pose, gripper)
        self.fake_pose = np.identity(4)

    @property
    def pose(self):
        """Read-only attribute that contains the current tcp world pose."""
        return self.fake_pose

    def is_pose_unsafe(self, pose_in_world) -> bool:
        """Check whether it is unsafe to go the given pose. Intended for use externally e.g. by a control of grasping
        strategy or internally as a safety check before executing a motion."""
        False

    def move_tcp(self, pose_in_world: np.ndarray):
        """Move the TCP to a pose with no guarantees on the path taken."""
        self.fake_pose = pose_in_world
        print(f"Moved TCP to: \n{self.fake_pose}")

    def move_tcp_linear(self, pose_in_world: np.ndarray, speed: float, acceleration: float):
        """Move the robot TCP linearly from its current pose to the specified pose."""
        self.fake_pose = pose_in_world
        print(f"Moved TCP linearly to: \n{self.fake_pose}")

    def move_tcp_linear_path(self, poses_in_world: List[np.ndarray], speed: float, acceleration: float):
        """Move the robot TCP linearly between waypoint poses."""
        self.fake_pose = poses_in_world[-1]
        print(f"Moved TCP along path to: \n{self.fake_pose}")

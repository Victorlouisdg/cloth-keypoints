"""Base classes for robotic hardware that serve two purposes:
    1) Standardize the interface for execution of movements on real hardware.
    2) Provide static and runtime information about the hardware setup to inform control decision.

    Currently 3 classes exist that related to each other as such:
    A DualArm setup has 2 RobotArms, a left one and a right one.
    A RobotArm optionally has a Gripper.

    RobotArm and Gripper have to be subclassed to be usable.
    DualArm can be used as is.

    All poses provided to these classes should be homogeneous (4, 4) numpy arrays.
"""
from abc import ABC, abstractmethod
from threading import Thread
from typing import List, Optional, Tuple

import numpy as np


class Gripper(ABC):
    """Simple interface to abstract phyiscal drivers (modbus/ TCP / ...)"""

    @abstractmethod
    def open(self):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def move_to_position(self, position=0.5):
        """position should a float between 0 and 1, where 0 is completely open and 1 is closed."""


class RobotArm(ABC):
    def __init__(
        self,
        name: str,
        robot_in_world_pose: np.ndarray,
        home_pose: np.ndarray,
        out_of_way_pose: np.ndarray,
        gripper: Optional[Gripper] = None,
    ):
        self.name = name
        self.robot_in_world_pose = robot_in_world_pose
        self.home_pose = home_pose
        self.out_of_way_pose = out_of_way_pose
        self.gripper = gripper

    @property
    def world_to_robot(self):
        return np.linalg.inv(self.robot_in_world_pose)

    @property
    @abstractmethod
    def pose(self):
        """Read-only attribute that contains the current tcp world pose."""

    @abstractmethod
    def is_pose_unsafe(self, pose_in_world) -> bool:
        """Check whether it is unsafe to go the given pose. Intended for use externally e.g. by a control of grasping
        strategy or internally as a safety check before executing a motion."""

    @abstractmethod
    def move_tcp(self, pose_in_world: np.ndarray):
        """Move the TCP to a pose with no guarantees on the path taken."""

    @abstractmethod
    def move_tcp_linear(self, pose_in_world: np.ndarray, speed: float, acceleration: float):
        """Move the robot TCP linearly from its current pose to the specified pose."""

    @abstractmethod
    def move_tcp_linear_path(self, poses_in_world: List[np.ndarray], speed: float, acceleration: float):
        """Move the robot TCP linearly between the waypoints of path."""


class DualArm:
    def __init__(self, left: RobotArm, right: RobotArm):
        self.left = left
        self.right = right

    @property
    def arms(self) -> Tuple[RobotArm, RobotArm]:
        return (self.left, self.right)

    def dual_gripper_open(self):
        self._execute_synchronously(
            self.left.gripper.open,
            self.right.gripper.open,
        )

    def dual_gripper_close(self):
        self._execute_synchronously(
            self.left.gripper.close,
            self.right.gripper.close,
        )

    def dual_gripper_move_to_position(self, position):
        self._execute_synchronously(
            self.left.gripper.move_to_position,
            self.right.gripper.move_to_position,
            (position,),
            (position,),
        )

    def dual_move_tcp(self, pose_in_world_left, pose_in_world_right):
        self._execute_synchronously(
            self.left.move_tcp,
            self.right.move_tcp,
            (pose_in_world_left,),
            (pose_in_world_right,),
        )

    def dual_move_tcp_linear(self, pose_in_world_left, pose_in_world_right, speed: float, acceleration: float):
        self._execute_synchronously(
            self.left.move_tcp_linear,
            self.right.move_tcp_linear,
            (pose_in_world_left, speed, acceleration),
            (pose_in_world_right, speed, acceleration),
        )

    def dual_move_tcp_linear_path(
        self,
        poses_in_world_left: List[np.ndarray],
        poses_in_world_right: List[np.ndarray],
        speed: float,
        acceleration: float,
    ):
        self._execute_synchronously(
            self.left.move_tcp_linear_path,
            self.right.move_tcp_linear_path,
            (poses_in_world_left, speed, acceleration),
            (poses_in_world_right, speed, acceleration),
        )

    @staticmethod
    def _execute_synchronously(func1, func2, args1: Tuple = (), args2: Tuple = ()):
        thread_1 = Thread(target=func1, args=args1)
        thread_2 = Thread(target=func2, args=args2)
        thread_1.start()
        thread_2.start()
        thread_2.join()
        thread_1.join()

from threading import Thread
from typing import List

import numpy as np
from rtde_control import RTDEControlInterface as RTDEControl
from scipy.spatial.transform import Rotation as R

from robotiq2f import Robotiq2F85TCP


def homogeneous_pose_to_position_and_rotvec(pose: np.ndarray):
    """converts a 4x4 homogeneous pose to [x,y,z, x_rot,y_rot,z_rot]"""
    position = pose[:3, 3]
    rpy = R.from_matrix(pose[:3, :3]).as_rotvec()
    return np.concatenate((position, rpy))


class Gripper:
    """Simple interface to abstract phyiscal drivers (modbus/ TCP / ...)"""

    def open(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError


class RobotiqTCP(Gripper):
    def __init__(self, robot_ip) -> None:
        super().__init__()
        self.gripper = Robotiq2F85TCP(robot_ip)
        self.gripper.activate_gripper()

    def open(self):
        return self.gripper.open()
        self.gripper.move_to_position()

    def close(self):
        return self.gripper.close()


class UR:
    DEFAULT_LINEAR_VEL = 0.1  # m/s
    DEFAULT_LINEAR_ACC = 0.4  # m/s^2
    DEFAULT_JOINT_VEL = 0.4  # rad/s
    DEFAULT_JOINT_ACC = 0.8  # rad/s^2
    DEFAULT_BLEND = 0.01
    """simple wrapper around the RTDE interface"""

    def __init__(self, ip: str, gripper: Gripper = None, robot_in_world_position=[0, 0, 0]):
        self.rtde = RTDEControl(ip)
        self.gripper = gripper

        self.robot_in_world_position = np.array(robot_in_world_position)
        self.home_pose = (
            None  # pose from which the robot can safely reach all manipulation poses and the out of way pose
        )
        self.out_of_way_pose = None  # pose used for capturing images without occlusions bc of robot

    def _transform_world_pose_to_robot_frame(self, pose_in_world):
        assert isinstance(pose_in_world, np.ndarray)
        assert len(pose_in_world) == 6, "poses should be <position, rotation vector>!"

        world_to_robot_translation = -self.robot_in_world_position
        pose_in_robot_frame = np.copy(pose_in_world)
        pose_in_robot_frame[:3] = pose_in_robot_frame[:3] + world_to_robot_translation
        return pose_in_robot_frame

    def check_is_not_necessarily_unsafe_pose(self, pose_in_robot_frame) -> None:

        unsafe = not self.rtde.isPoseWithinSafetyLimits(pose_in_robot_frame)
        if unsafe:
            raise ValueError(f"this pose:{pose_in_robot_frame} is not reachable!")

        unsafe = np.linalg.norm(pose_in_robot_frame[:3]) < 0.1
        if unsafe:
            raise ValueError(f"this pose:{pose_in_robot_frame} would most likely lead to a collision!")

    def is_world_pose_reachable(self, pose_in_world_frame: np.ndarray) -> bool:
        robot_pose = self._transform_world_pose_to_robot_frame(pose_in_world_frame)
        return self.rtde.isPoseWithinSafetyLimits(robot_pose)

    def moveL(self, pose_in_world_frame: np.array, vel=DEFAULT_LINEAR_VEL, acc=DEFAULT_LINEAR_ACC):
        pose_in_robot_frame = self._transform_world_pose_to_robot_frame(pose_in_world_frame)
        self.check_is_not_necessarily_unsafe_pose(pose_in_robot_frame)
        self.rtde.moveL(pose_in_robot_frame, vel, acc)

    def moveP(self, trajectory_in_world_frame: List[np.array]):
        trajectory_in_robot_frame = trajectory_in_world_frame
        for waypoint in trajectory_in_robot_frame:
            waypoint[:6] = self._transform_world_pose_to_robot_frame(waypoint[:6])
            self.check_is_not_necessarily_unsafe_pose(waypoint[:6])
        self.rtde.moveL(trajectory_in_robot_frame)

    def moveJ_IK(self, pose_in_world_frame: np.array, vel=DEFAULT_JOINT_VEL, acc=DEFAULT_JOINT_ACC):
        pose_in_robot_frame = self._transform_world_pose_to_robot_frame(pose_in_world_frame)
        self.check_is_not_necessarily_unsafe_pose(pose_in_robot_frame)
        self.rtde.moveJ_IK(pose_in_robot_frame, vel, acc)


class DualArmUR:
    DEFAULT_LINEAR_VEL = UR.DEFAULT_LINEAR_VEL
    DEFAULT_LINEAR_ACC = UR.DEFAULT_LINEAR_ACC
    DEFAULT_JOINT_VEL = UR.DEFAULT_JOINT_VEL
    DEFAULT_JOINT_ACC = UR.DEFAULT_JOINT_ACC
    DEFAULT_BLEND = UR.DEFAULT_BLEND
    """
    Idea of this class is mainly to enable synchronous execution of motions on both robots in the dual-arm setup.
    e.g. you want to move the robots each to a pose and wait untill they have both arrived.
    If the motions are similar for each robot, they will also approx. execute them synchronously (not only wait untill both are finished)
    which can be used for example to fold cloth with 2 arms.
    """

    def __init__(self, victor: UR, louise: UR) -> None:
        self.victor_ur = victor
        self.louise_ur = louise

    def dual_moveL(
        self,
        pose_in_world_victor: np.ndarray,
        pose_in_world_louise: np.ndarray,
        vel=DEFAULT_LINEAR_VEL,
        acc=DEFAULT_LINEAR_ACC,
    ):
        self._dual_arm_sync_exec(
            self.victor_ur.moveL,
            self.louise_ur.moveL,
            (pose_in_world_victor, vel, acc),
            (pose_in_world_louise, vel, acc),
        )

    def dual_moveP(self, trajectory_in_world_victor: List[np.ndarray], trajectory_in_world_louise: List[np.ndarray]):
        self._dual_arm_sync_exec(
            self.victor_ur.moveP, self.louise_ur.moveP, (trajectory_in_world_victor,), (trajectory_in_world_louise,)
        )

    def dual_moveJ_IK(
        self,
        pose_in_world_victor: np.ndarray,
        pose_in_world_louise: np.ndarray,
        vel=DEFAULT_JOINT_VEL,
        acc=DEFAULT_JOINT_ACC,
    ):
        self._dual_arm_sync_exec(
            self.victor_ur.moveJ_IK,
            self.louise_ur.moveJ_IK,
            (pose_in_world_victor, vel, acc),
            (pose_in_world_louise, vel, acc),
        )

    @staticmethod
    def _dual_arm_sync_exec(func1, func2, args1, args2):
        thread_1 = Thread(target=func1, args=args1)
        thread_2 = Thread(target=func2, args=args2)
        thread_1.start()
        thread_2.start()
        thread_2.join()
        thread_1.join()

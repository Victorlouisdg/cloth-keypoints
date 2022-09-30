from pickletools import unicodestring1
from threading import Thread
from typing import List

import numpy as np
from robotiq2f_tcp import Robotiq2F85TCP
from rtde_control import RTDEControlInterface as RTDEControl
from scipy.spatial.transform import Rotation as R

DEFAULT_LINEAR_VEL = 0.1
DEFAULT_LINEAR_ACC = 0.1
DEFAULT_JOINT_VEL = 0.3
DEFAULT_JOINT_ACC = 0.1


class Gripper:
    """Simple interface to abstract phyiscal drivers (modbus/ TCP / ...)"""

    def open(self):
        raise NotImplemented

    def close(self):
        raise NotImplemented


class RobotiqTCP(Gripper):
    def __init__(self, robot_ip) -> None:
        super().__init__()
        self.gripper = Robotiq2F85TCP(robot_ip)
        self.gripper.activate_gripper()

    def open(self):
        return self.gripper.open()

    def close(self):
        return self.gripper.close()


class RobotiqUSB(Gripper):
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError


class UR:
    """simple wrapper around the RTDE interface"""

    def __init__(self, ip: str, gripper: Gripper = None):
        self.rtde = RTDEControl(ip)
        self.gripper = gripper

    def moveL(self, waypoint: List, vel=DEFAULT_LINEAR_VEL, acc=DEFAULT_LINEAR_ACC):
        # checks
        self.rtde.moveL(waypoint, vel, acc)

    def moveP(self, waypoints: List[List]):
        # check
        self.rtde.moveL(waypoints)

    def moveJ_IK(self, waypoint: List, vel=DEFAULT_JOINT_VEL, acc=DEFAULT_JOINT_ACC):
        # checks
        self.rtde.moveJ_IK(waypoint, vel, acc)


class DualArmUR:
    """
    Idea of this class is mainly to enable synchronous execution of motions on both robots in the dual-arm setup.
    e.g. you want to move the robots each to a pose and wait untill they have both arrived.
    If the motions are similar for each robot, they will also approx. execute them synchronously (not only wait untill both are finished)
    which can be used for example to fold cloth with 2 arms.
    """

    def __init__(self, victor: UR, louise: UR) -> None:
        self.victor_ur = victor
        self.louise_ur = louise

    def dual_moveL(self, waypoint_victor: List, waypoint_louise, vel=DEFAULT_LINEAR_VEL, acc=DEFAULT_LINEAR_ACC):
        self._dual_arm_sync_exec(
            self.victor_ur.moveL, self.louise_ur.moveL, (waypoint_victor, vel, acc), (waypoint_louise, vel, acc)
        )

    def dual_moveP(self, waypoints_victor: List[List], waypoints_louise: List[List]):
        self._dual_arm_sync_exec(self.victor_ur.moveP, self.louise_ur.moveP, waypoints_victor, waypoints_louise)

    def dual_moveJ_IK(
        self, waypoint_victor: List, waypoint_louise: List, vel=DEFAULT_JOINT_VEL, acc=DEFAULT_JOINT_ACC
    ):
        self._dual_arm_sync_exec(
            self.victor_ur.moveJ_IK, self.louise_ur.moveJ_IK, (waypoint_victor, vel, acc), (waypoint_louise, vel, acc)
        )

    @staticmethod
    def _dual_arm_sync_exec(func1, func2, args1, args2):
        thread_1 = Thread(target=func1, args=args1)
        thread_2 = Thread(target=func2, args=args2)
        thread_1.start()
        thread_2.start()
        thread_2.join()
        thread_1.join()

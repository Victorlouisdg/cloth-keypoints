"""
Functions that return prebuilt dual arm setups for specific scenarios.
"""

import numpy as np
from cloth_manipulation.hardware.base_classes import DualArm
from cloth_manipulation.hardware.fake_hardware import FakeArm, FakeGripper
from cloth_manipulation.hardware.robotiq2f_gripper import Robotiq2F85
from cloth_manipulation.hardware.universal_robots import UR


def setup_victor_louise() -> DualArm:
    """Sets up the robots for towel folding on the Victor-Louise UR3e cart setup.
    Both robots are on the X-axis, with manually measured offsets of + (victor) and -(louise) 39cm.
    """
    ip_victor = "10.42.0.162"
    ip_louise = "10.42.0.163"

    gripper_victor = Robotiq2F85(ip_victor)
    gripper_louise = Robotiq2F85(ip_louise)

    victor_in_world = np.identity(4)
    victor_in_world[0, -1] -= 0.39

    louise_in_world = np.identity(4)
    louise_in_world[0, -1] += 0.39

    home_victor = victor_in_world.copy()
    home_victor[:3, -1] += [0.2, -0.1, 0.2]

    home_louise = louise_in_world.copy()
    home_louise[:3, -1] += [-0.2, -0.1, 0.2]

    # home_orientation = list(R.from_euler("yz", [np.pi, -np.pi / 2]).as_rotvec())

    out_of_way_victor = victor_in_world.copy()
    out_of_way_victor[:3, -1] += [-0.05, -0.2, 0.2]

    out_of_way_louise = louise_in_world.copy()
    out_of_way_louise[:3, -1] += [0.05, -0.2, 0.2]

    victor = UR("victor", victor_in_world, gripper_victor, ip_victor, home_victor, out_of_way_victor)
    louise = UR("louise", louise_in_world, gripper_louise, ip_louise, home_louise, out_of_way_louise)

    victor_louise = DualArm(left=victor, right=louise)
    return victor_louise


def setup_fake_victor_louise() -> DualArm:
    gripper_victor = FakeGripper()
    gripper_louise = FakeGripper()

    victor_in_world = np.identity(4)
    victor_in_world[0, -1] -= 0.39

    louise_in_world = np.identity(4)
    louise_in_world[0, -1] += 0.39

    home_victor = victor_in_world.copy()
    home_victor[:3, -1] += [0.2, -0.1, 0.2]

    home_louise = louise_in_world.copy()
    home_louise[:3, -1] += [-0.2, -0.1, 0.2]

    out_of_way_victor = victor_in_world.copy()
    out_of_way_victor[:3, -1] += [-0.05, -0.2, 0.2]

    out_of_way_louise = louise_in_world.copy()
    out_of_way_louise[:3, -1] += [0.05, -0.2, 0.2]

    victor = FakeArm("victor", victor_in_world, home_victor, out_of_way_victor, gripper_victor)
    louise = FakeArm("louise", louise_in_world, home_louise, out_of_way_louise, gripper_louise)

    victor_louise = DualArm(left=victor, right=louise)
    return victor_louise

import numpy as np
from cloth_manipulation.controllers import ReorientTowelController
from cloth_manipulation.geometry import move_closer
from cloth_manipulation.hardware.setup_hardware import setup_fake_victor_louise
from cloth_manipulation.motion_primitives.fold_execution import (
    execute_dual_fold_trajectories,
    execute_single_fold_trajectory,
)
from cloth_manipulation.motion_primitives.fold_trajectory_parameterization import CircularFoldTrajectory
from cloth_manipulation.motion_primitives.pull import TowelReorientPull, execute_pull_primitive

np.set_printoptions(suppress=True)
np.set_printoptions(precision=2)

victor_louise = setup_fake_victor_louise()

for arm in victor_louise.arms:
    print(f"====== {arm.name} ======")
    print(f"Pose of arm base: \n{arm.robot_in_world_pose}")
    origin_pose = np.identity(4)
    arm.move_tcp(origin_pose)


print("Starting fake pull.")
towel_length = 0.3
towel_width = 0.2
x = towel_width / 2
y = towel_length / 2
dx = x  # 0.02
dy = y  # 0.05
shift = np.array([dx, dy, 0])

corners_centered = np.array(
    [
        np.array([x, y, 0]),
        np.array([-x, y, 0]),
        np.array([x, -y, 0]),
        np.array([-x, -y, 0]),
    ]
)
corners_shifted = corners_centered + shift

print("=== Testing pull execution ===")
print(f"Towel corners : \n{corners_centered}")

pullprimitive = TowelReorientPull(corners_shifted, victor_louise)
print(pullprimitive)
execute_pull_primitive(pullprimitive, victor_louise)

print("=== Testing dual arm fold ===")
print(f"Towel corners : \n{corners_centered}")

end_louise, end_victor, start_victor, start_louise = corners_centered

end_louise, end_victor = move_closer(end_louise, end_victor, 0.04)
start_louise, start_victor = move_closer(start_louise, start_victor, 0.04)

fold_trajectory_victor = CircularFoldTrajectory(start_victor, end_victor)
fold_trajectory_louise = CircularFoldTrajectory(start_louise, end_louise)

execute_dual_fold_trajectories(fold_trajectory_victor, fold_trajectory_louise, victor_louise)


print("=== Testing single arm fold  ===")
execute_single_fold_trajectory(fold_trajectory_victor, victor_louise.left)


print("=== Testing reorient pull controller ===")

corners_shifted

controller = ReorientTowelController(victor_louise)

corners = corners_shifted
while not controller.finished:
    controller.act(corners)
    corners = corners_centered  # Imitate execution of pull

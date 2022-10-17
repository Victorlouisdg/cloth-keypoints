import airo_blender_toolkit as abt
import numpy as np
from cloth_manipulation.geometry import top_down_orientation
from mathutils import Vector
from scipy.spatial.transform import Rotation


def tilted_pull_orientation(pull_location, robot_location, tilt_angle=45):
    robot_to_pull = pull_location - robot_location
    if np.linalg.norm(robot_to_pull) < 0.3:
        tilt_angle = -tilt_angle  # tilt inwards

    gripper_open_direction = robot_to_pull
    top_down = top_down_orientation(gripper_open_direction)

    gripper_y = top_down[:, 1]
    rotation = Rotation.from_rotvec(np.deg2rad(tilt_angle) * gripper_y)

    gripper_orienation = rotation.as_matrix() @ top_down

    return gripper_orienation


def visualize_robots_base_plates(robots_center=(0, 0, 0)):
    robots_center = Vector(robots_center)
    robots_x = 0.39
    for dx in [robots_x, -robots_x]:
        base = abt.Cube(size=0.14, location=robots_center + Vector([dx, 0, 0]), scale=(1, 1, 0.1))
        base.add_colored_material([0.25, 0.25, 0.25, 1.000000])


abt.clear_scene()
visualize_robots_base_plates()

robot_location = np.array([-0.39, 0, 0])
max_theta = 8 * np.pi
for theta in np.arange(0, max_theta, np.pi / 10):
    r = 0.1 + 0.4 * (theta / max_theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pull_location = robot_location + np.array([x, y, 0])
    orientation = tilted_pull_orientation(pull_location, robot_location)
    frame = abt.Frame.from_orientation_and_position(orientation, pull_location)
    abt.visualize_transform(frame, scale=0.05)

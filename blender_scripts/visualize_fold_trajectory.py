from typing import List
import airo_blender_toolkit as abt
from cloth_manipulation.motion_primitives.fold_trajectory_parameterization import CircularFoldTrajectory
from cloth_manipulation.utils import get_ordered_keypoints
import numpy as np
from mathutils import Vector
import argparse
import sys


def default_towel() -> abt.Towel:
    towel_length = 0.3
    towel_width = 0.2
    towel = abt.Towel(towel_length, towel_width)
    # towel.location = 0.02, 0.05, 0.0
    # towel.location = 0.0, 0.3, 0.0
    towel.location = 0.3, 0.3, 0.0
    # towel.location = towel_width / 2, -towel_length / 2, 0.0
    towel.rotation_euler = 0, 0, np.pi / 16
    # towel.rotation_euler = 0, 0, np.pi / 2
    towel.apply_transforms()
    return towel


def random_towel(seed) -> abt.Towel:
    np.random.seed(seed)
    towel_length = np.random.uniform(0.2, 0.7)
    towel_width = np.random.uniform(towel_length / 2, towel_length)
    towel = abt.Towel(towel_length, towel_width)
    towel.location = *np.random.uniform(-0.1, 0.1, size=2), 0.0
    towel.rotation_euler = [0, 0, np.random.uniform(-np.pi / 16, np.pi / 16)]
    max_noise = towel_width / 20
    for _ in range(4):
        towel.blender_object.data.vertices[0].co += Vector([*np.random.uniform(-max_noise, max_noise, 2), 0])
    towel.apply_transforms()
    return towel


def visualize_robots_base_plates(robots_center=(0, 0, 0)):
    robots_center = Vector(robots_center)
    robots_x = 0.39
    for dx in [robots_x, -robots_x]:
        base = abt.Cube(size=0.14, location=robots_center + Vector([dx, 0, 0]), scale=(1, 1, 0.1))
        base.add_colored_material([0.25, 0.25, 0.25, 1.000000])


def visualize_fold_trajectory(corners):
    corners = get_ordered_keypoints(corners)
    end_louise, end_victor, start_victor, start_louise = corners
    fold_trajectory_victor = CircularFoldTrajectory(start_victor, end_victor)
    fold_trajectory_louise = CircularFoldTrajectory(start_louise, end_louise)

    visualization_objects = []
    for transform in fold_trajectory_victor.get_fold_path(8):
        empty = abt.visualize_transform(transform)
        visualization_objects.append(empty)


    for transform in fold_trajectory_louise.get_fold_path(8):
        empty = abt.visualize_transform(transform, scale=0.05)
        visualization_objects.append(empty)

    return visualization_objects


if __name__ == "__main__":
    abt.clear_scene()
    if "--" not in sys.argv:
        # Default towel
        visualize_robots_base_plates()
        towel = default_towel()
        keypoints = {"corner": list(towel.keypoints_3D.values())}
        corners = keypoints["corner"]
        visualize_fold_trajectory(corners)
    else:
        argv = sys.argv[sys.argv.index("--") + 1 :]
        parser = argparse.ArgumentParser()
        parser.add_argument("amount_of_towels", type=int)
        args = parser.parse_known_args(argv)[0]

        amount_of_towels = args.amount_of_towels
        columns = int(np.ceil(np.sqrt(amount_of_towels)))
        shift = 1.2

        for i in range(amount_of_towels):
            print(f"Starting towel {i}")
            x = shift * (i % columns)
            y = -shift * (i // columns)
            offset = Vector((x, y, 0))

            sphere = abt.Sphere(location=offset, radius=0.005)
            sphere.add_colored_material((0,0,1,1))

            visualize_robots_base_plates(offset)
            towel = random_towel(i)
            keypoints = {"corner": list(towel.keypoints_3D.values())}
            corners = keypoints["corner"]
            visualization_objects = visualize_fold_trajectory(corners)

            towel.location += offset
            for object in visualization_objects:
                object.location += offset

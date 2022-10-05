from typing import List
import airo_blender_toolkit as abt
import numpy as np
from mathutils import Vector
from cloth_manipulation.motion_primitives.pull import TowelReorientPull
import argparse
import sys


def default_towel() -> abt.Towel:
    towel_length = 0.3
    towel_width = 0.2
    towel = abt.Towel(towel_length, towel_width)
    towel.location = 0.02, 0.05, 0.0
    towel.location = towel_width / 2, -towel_length / 2, 0.0
    towel.rotation_euler = 0, 0, np.pi / 16
    towel.apply_transforms()
    return towel


def random_towel(seed) -> abt.Towel:
    np.random.seed(seed)
    towel_length = np.random.uniform(0.2, 0.7)
    towel_width = np.random.uniform(towel_length / 2, towel_length)
    towel = abt.Towel(towel_length, towel_width)
    towel.location = *np.random.uniform(-0.1, 0.1, size=2), 0.0
    towel.rotation_euler = [0, 0, np.random.uniform(0.0, 2 * np.pi)]
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


def visualize_pull_primitve(corners):
    pullprimitive = TowelReorientPull(corners)
    start_pose = pullprimitive.get_pull_start_pose()
    end_pose = pullprimitive.get_pull_end_pose()

    visualization_objects = []
    empty_start = abt.visualize_transform(start_pose)
    empty_end = abt.visualize_transform(end_pose)
    visualization_objects += [empty_start, empty_end]

    for i in range(4):
        sphere = abt.Sphere(location=pullprimitive.desired_corners[i], radius=0.01)
        sphere.add_colored_material([0, 1, 0, 1])
        sphere.blender_object.name = f"desired_corner_{i}"
        visualization_objects.append(sphere)

    for corner, destination in pullprimitive.corner_destinations:
        line = abt.visualize_line_segment(corner, destination, thickness=0.002, color=abt.colors.light_blue)
        visualization_objects.append(line)

    selected_line = abt.visualize_line_segment(
        pullprimitive.start_original, pullprimitive.end_original, color=(0, 1, 0, 1)
    )
    visualization_objects.append(selected_line)

    return visualization_objects


if __name__ == "__main__":
    abt.clear_scene()
    if "--" not in sys.argv:
        # Default towel
        visualize_robots_base_plates()
        towel = default_towel()
        keypoints = {"corner": list(towel.keypoints_3D.values())}
        corners = keypoints["corner"]
        visualize_pull_primitve(corners)
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
            visualization_objects = visualize_pull_primitve(corners)

            towel.location += offset
            for object in visualization_objects:
                object.location += offset

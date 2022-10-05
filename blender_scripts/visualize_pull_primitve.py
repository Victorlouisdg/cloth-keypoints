from typing import List
import airo_blender_toolkit as abt
import numpy as np
from mathutils import Vector
from cloth_manipulation.motion_primitives.pull import select_towel_pull

abt.clear_scene()

robots_x = 0.39
abt.Cube(size=0.14, location=(robots_x, 0, 0), scale=(1, 1, 0.1))
abt.Cube(size=0.14, location=(-robots_x, 0, 0), scale=(1, 1, 0.1))


# Set up example towel
towel_length = 0.3
towel_width = 0.2
towel = abt.Towel(towel_length, towel_width)
# towel.blender_object.data.vertices[0].co += Vector([0.05, 0.1, 0])
# towel.location = 0.02, 0.05, 0.0
towel.location = towel_width/2, -towel_length/2, 0.0
# towel.rotation_euler = 0, 0, np.pi / 16

# Random towel
# np.random.seed(2)
# towel_length = np.random.uniform(0.3, 1.0)
# towel_width = np.random.uniform(towel_length / 2, towel_length)
# towel = abt.Towel(towel_length, towel_width)
# towel.location = *np.random.uniform(-0.1, 0.1, size=2), 0.0
# towel.rotation_euler = [0, 0, np.random.uniform(0.0, 2 * np.pi)]
# for i in range(4):
#     towel.blender_object.data.vertices[0].co += Vector([np.random.uniform(0.0, 0.05), np.random.uniform(0.0, 0.05), 0])


towel.apply_transforms()

# Anonymize keypoint as detector can't differentiate between them.
keypoints = {"corner": list(towel.keypoints_3D.values())}
corners = keypoints["corner"]

corners = [np.array([0.0,0.0,0.01]),np.array([0.23,0.0,0.01]),np.array([0.23,-0.3,0.01]),np.array([0.0,-0.3,0.01])]

pullprimitive = select_towel_pull(corners)
start_pose = pullprimitive.get_pull_start_pose()
end_pose = pullprimitive.get_pull_end_pose()

abt.visualize_transform(start_pose)
abt.visualize_transform(end_pose)

test_keypoints = [np.array([0.0,0.0,0.01]),np.array([0.23,0.0,0.01]),np.array([0.23,-0.3,0.01]),np.array([0.0,-0.3,0.01])]
for keypoint in test_keypoints:
    abt.Sphere(location=keypoint, radius=0.01)
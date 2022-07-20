import argparse
import os
import random
import sys

import airo_blender_toolkit as abt
import bpy
import numpy as np

os.environ["INSIDE_OF_THE_INTERNAL_BLENDER_PYTHON_ENVIRONMENT"] = "1"
import blenderproc as bproc  # noqa


def generate_towel():
    towel_length = np.random.uniform(0.4, 0.7)
    towel_width = np.random.uniform(0.2, towel_length)
    towel = abt.Towel(towel_length, towel_width)
    return towel


def generate_shirt():
    shirt = abt.PolygonalShirt()
    return shirt


def generate_pants():
    pants = abt.PolygonalPants()
    return pants


def generate_scene(seed, generate_cloth_fn=generate_towel):
    print(seed)
    os.environ["BLENDER_PROC_RANDOM_SEED"] = str(seed)
    os.getenv("BLENDER_PROC_RANDOM_SEED")
    bproc.init()

    # renderer settings
    bpy.context.scene.cycles.adaptive_threshold = 0.2
    bpy.context.scene.cycles.use_denoising = False

    haven_folder = os.path.join(abt.assets_path(), "haven")
    haven_textures_folder = os.path.join(haven_folder, "textures")

    # create ground texture plane
    ground = bproc.object.create_primitive("PLANE")
    ground.blender_obj.name = "Ground"
    ground.set_scale([12] * 3)
    ground.set_rotation_euler([0, 0, np.random.uniform(0.0, 2 * np.pi)])
    ground_texture = abt.random_texture_name(haven_textures_folder)
    print(ground_texture)
    bproc.api.loader.load_haven_mat(haven_textures_folder, [ground_texture])
    ground_material = bpy.data.materials[ground_texture]
    ground.blender_obj.data.materials.append(ground_material)

    # Temporary way to make textures look smaller
    mesh = ground.blender_obj.data
    uv_layer = mesh.uv_layers.active
    for loop in mesh.loops:
        uv_layer.data[loop.index].uv *= 12

    generate_cloth = random.choice([generate_towel, generate_shirt, generate_pants])
    cloth = generate_cloth()

    cloth.set_rotation_euler([0, 0, np.random.uniform(0.0, 2 * np.pi)])
    # shift towel in view
    x_shift = float(np.random.uniform(-0.1, 0.1))
    y_shift = float(np.random.uniform(-0.1, 0.1))

    cloth.set_location((x_shift, y_shift, 0.001))
    cloth.persist_transformation_into_mesh()

    cloth_material = cloth.new_material("Cloth")
    cloth_material.set_principled_shader_value("Base Color", abt.random_hsv())

    # add camera
    camera = bpy.context.scene.camera
    camera_location = bproc.sampler.part_sphere(center=[0, 0, 0], radius=1.0, mode="INTERIOR", dist_above_center=0.85)
    camera_rotation = bproc.python.camera.CameraUtility.rotation_from_forward_vec((0, 0, 0) - camera_location)
    camera_pose = bproc.math.build_transformation_mat(camera_location, camera_rotation)
    bproc.camera.add_camera_pose(camera_pose)

    camera.scale = [0.2] * 3  # blender camera object size (no effect on generated images)
    camera.data.lens = 28  # focal distance [mm] - fov approx

    # add HDRI
    hdri_path = bproc.loader.get_random_world_background_hdr_img_path_from_haven(haven_folder)
    hdri_rotation = np.random.uniform(0, 2 * np.pi)
    abt.load_hdri(hdri_path, hdri_rotation)

    # shader graph
    tree = bpy.data.materials["Cloth"].node_tree
    cloth_shader = tree.nodes["Principled BSDF"]
    cloth_shader.inputs["Roughness"].default_value = 1.0

    n_random_objects = int(np.random.uniform(0.0, 5.0))
    for i in range(n_random_objects):
        print(i)
        abt.load_thingi10k_object()

    return cloth


if __name__ == "__main__":
    seed = 2022
    if "--" in sys.argv:
        argv = sys.argv[sys.argv.index("--") + 1 :]
        parser = argparse.ArgumentParser()
        parser.add_argument("seed", type=int)
        args = parser.parse_known_args(argv)[0]
        seed = args.seed
    cloth = generate_scene(seed)
    cloth.visualize_keypoints()

    print("done")

import numpy as np
from camera_toolkit.reproject import reproject_to_world_z_plane
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.manual_keypoints import ClothTransform
from cloth_manipulation.motion_primitives.pull import execute_pull_primitive, select_towel_pull
from cloth_manipulation.setup_hw import setup_hw

if __name__ == "__main__":
    dual_arm = setup_hw()
    from camera_toolkit.reproject import reproject_to_world_z_plane
    from cloth_manipulation.manual_keypoints import aruco_in_camera_transform, get_manual_keypoints

    # open ZED (and assert it is available)
    Zed2i.list_camera_serial_numbers()
    zed = Zed2i()

    # L move to home to avoid collisions with other robot?
    dual_arm.dual_moveL(
        dual_arm.victor_ur.home_pose, dual_arm.louise_ur.home_pose, vel=2 * dual_arm.DEFAULT_LINEAR_VEL
    )
    while True:
        dual_arm.dual_moveL(
            dual_arm.victor_ur.out_of_way_pose, dual_arm.louise_ur.out_of_way_pose, vel=2 * dual_arm.DEFAULT_LINEAR_VEL
        )
        img = zed.get_rgb_image()
        transformed_image = ClothTransform.transform_image(img)

        transformed_cv_img = Zed2i.image_shape_torch_to_opencv(transformed_image)
        transformed_keypoints = np.array(get_manual_keypoints(transformed_cv_img, 4))

        keypoints_in_camera = ClothTransform.reverse_transform_keypoints(transformed_keypoints)
        keypoints_in_world = reproject_to_world_z_plane(
            keypoints_in_camera, zed.get_camera_matrix(), aruco_in_camera_transform
        )

        pullprimitive = select_towel_pull(keypoints_in_world)
        if np.linalg.norm(pullprimitive.start_position - pullprimitive.end_position) < 0.05:
            print("pull was less than 5cm, no need to execute")
            break
        print(pullprimitive)
        execute_pull_primitive(pullprimitive, dual_arm)

    dual_arm.dual_moveL(
        dual_arm.victor_ur.home_pose, dual_arm.louise_ur.home_pose, vel=2 * dual_arm.DEFAULT_LINEAR_VEL
    )

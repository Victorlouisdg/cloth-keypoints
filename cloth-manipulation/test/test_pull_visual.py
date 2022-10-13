"""Script that shows that output of a keypoint detector from wandb on a ZED camera feed."""
from typing import List, Tuple
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.fake_robots import FakeDualArm, FakeRobot
import cv2
import pyzed.sl as sl
from cloth_manipulation.manual_keypoints import ClothTransform
import numpy as np
import wandb
from pathlib import Path
import torch
import pickle
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.models.backbones.convnext_unet import ConvNeXtUnet
from keypoint_detection.utils.visualization import overlay_image_with_heatmap
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap
from camera_toolkit.reproject import reproject_to_world_z_plane
from cloth_manipulation.motion_primitives.pull import TowelReorientPull
import cloth_manipulation.camera_mapping as cm

def get_wandb_model(checkpoint_reference, backbone=ConvNeXtUnet()):
    """
    checkpoint_reference: str e.g. 'airo-box-manipulation/iros2022_0/model-17tyvqfk:v3'
    """
    # download checkpoint locally (if not already cached)
    run = wandb.init(project="inference", entity="airo-box-manipulation")
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()
    model_path = Path(artifact_dir) / "model.ckpt"
    model = KeypointDetector.load_from_checkpoint(model_path, backbone=backbone)
    return model

def draw_cloth_transform_rectangle(image_full_size) -> np.ndarray:
    u_top = ClothTransform.crop_start_u
    u_bottom = u_top + ClothTransform.crop_width
    v_top = ClothTransform.crop_start_v
    v_bottom = v_top + ClothTransform.crop_height

    top_left = (u_top, v_top)
    bottom_right = (u_bottom, v_bottom)

    image = cv2.rectangle(
        image_full_size, top_left, bottom_right, (255, 0, 0), thickness=2
    )
    return image


def insert_transformed_into_original(original, transformed):
    u_top = ClothTransform.crop_start_u
    u_bottom = u_top + ClothTransform.crop_width
    v_top = ClothTransform.crop_start_v
    v_bottom = v_top + ClothTransform.crop_height

    transformed_unresized = cv2.resize(
        transformed, (ClothTransform.crop_width, ClothTransform.crop_height)
    )
    original[
        v_top:v_bottom,
        u_top:u_bottom,
    ] = transformed_unresized


resolution = sl.RESOLUTION.HD720

zed = Zed2i(resolution=resolution, serial_number=cm.CameraMapping.serial_top)
image = zed.get_rgb_image()
image = zed.image_shape_torch_to_opencv(image)

print(zed.get_camera_matrix().shape)
print(zed.get_camera_matrix())

print("image_shape", image.shape)
h, w, c = image.shape

new_size = 600
ClothTransform.crop_start_u = (w - new_size) // 2
ClothTransform.crop_width = new_size
ClothTransform.crop_start_v = (h - new_size) // 2
ClothTransform.crop_height = new_size


checkpoint_reference = "airo-box-manipulation/iros2022/model-wy78a0el:v3"  # synthetic
# checkpoint_reference = 'airo-box-manipulation/iros2022/model-1h8f5ldx:v12'
keypoint_detector = get_wandb_model(checkpoint_reference)

# load camera to marker transform
with open(Path(__file__).parent.parent / "cloth_manipulation" / "marker.pickle", "rb") as f:
    aruco_in_camera_position, aruco_in_camera_orientation = pickle.load(f)
# get camera extrinsics transform
aruco_in_camera_transform = np.eye(4)
aruco_in_camera_transform[:3, :3] = aruco_in_camera_orientation
aruco_in_camera_transform[:3, 3] = aruco_in_camera_position


print("Press q to quit.")

while True:
    image = zed.get_rgb_image()
    transformed_image = ClothTransform.transform_image(image)
    image_batched = torch.Tensor(transformed_image).unsqueeze(0) / 255.0

    with torch.no_grad():
        heatmap = keypoint_detector(image_batched)

    heatmap_channel_batched = heatmap.squeeze(1)
    heatmap_channel = heatmap_channel_batched.squeeze(0)

    overlayed = overlay_image_with_heatmap(image_batched, heatmap_channel_batched)
    overlayed = overlayed.squeeze(0).numpy()
    overlayed = zed.image_shape_torch_to_opencv(overlayed)
    overlayed = overlayed.copy()

    keypoints = get_keypoints_from_heatmap(
        heatmap_channel.cpu(), min_keypoint_pixel_distance=4, max_keypoints=4
    )


    keypoints_in_camera = ClothTransform.reverse_transform_keypoints(np.array(keypoints))
    keypoints_in_world = reproject_to_world_z_plane(
        keypoints_in_camera, zed.get_camera_matrix(), aruco_in_camera_transform
    )

    victor = FakeRobot(robot_in_world_position=[-0.4,0,0])
    louise = FakeRobot(robot_in_world_position=[0.4,0,0])
    dual_arm = FakeDualArm(victor, louise)
   
    def project_to_camera_plane(point_3D: List[float]) -> Tuple[int, int]:
        point_homogeneous = np.ones((4,1))
        point_homogeneous[:3, 0] = point_3D
        world_to_camera = aruco_in_camera_transform
        point_camera = world_to_camera @ point_homogeneous
        point_3D = point_camera[:3, 0]
        point_camera_2D = zed.get_camera_matrix() @ point_3D
        point_2D = point_camera_2D[:2] / point_camera_2D[2]
        # print(point_3D, point_camera_2D, point_2D)
        u, v = point_2D
        return int(u), int(v)
 

 

    for keypoint in keypoints:
        overlayed = cv2.circle(overlayed, keypoint, 6, (0, 1, 0))

    overlayed *= 255.0
    overlayed = overlayed.astype(np.uint8)  # copy()

    image = zed.image_shape_torch_to_opencv(image)
    image = image.copy()
    insert_transformed_into_original(image, overlayed)

    if len(keypoints) == 4:
        pullprimitive = TowelReorientPull(keypoints_in_world, dual_arm)
        start = project_to_camera_plane(pullprimitive.start_original)
        end = project_to_camera_plane(pullprimitive.end_original)
        image = cv2.circle(image, start, 10, (255, 0, 0), thickness=2)
        image = cv2.circle(image, end, 10, (0, 0, 255), thickness=2)

    print(zed.get_camera_matrix())
    origin = project_to_camera_plane([0.0, 0.0, 0.0])
    print(origin)
    image = cv2.circle(image, origin, 10, (0, 255, 255), thickness=2)



    image = draw_cloth_transform_rectangle(image)

    cv2.imshow("Image", image)
    key = cv2.waitKey(10)
    if key == ord("q"):
        cv2.destroyAllWindows()
        zed.close()
        break

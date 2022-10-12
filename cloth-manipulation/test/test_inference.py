"""Script that shows that output of a keypoint detector from wandb on a ZED camera feed."""
from camera_toolkit.zed2i import Zed2i
import cv2
import pyzed.sl as sl
from cloth_manipulation.manual_keypoints import ClothTransform
import numpy as np
import wandb
from pathlib import Path
import torch
from keypoint_detection.models.detector import KeypointDetector
from keypoint_detection.models.backbones.convnext_unet import ConvNeXtUnet
from keypoint_detection.utils.visualization import overlay_image_with_heatmap
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap


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


serial_numbers = {
    "top": 38633712,
    "side": 35357320,
    "front": 31733653,
}


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

zed = Zed2i(resolution=resolution, serial_number=serial_numbers["top"])
image = zed.get_rgb_image("right")
image = zed.image_shape_torch_to_opencv(image)

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
    for keypoint in keypoints:
        overlayed = cv2.circle(overlayed, keypoint, 6, (0, 1, 0))

    overlayed *= 255.0
    overlayed = overlayed.astype(np.uint8)  # copy()

    image = zed.image_shape_torch_to_opencv(image)
    image = image.copy()
    insert_transformed_into_original(image, overlayed)
    image = draw_cloth_transform_rectangle(image)

    cv2.imshow("Image", image)
    key = cv2.waitKey(10)
    if key == ord("q"):
        cv2.destroyAllWindows()
        zed.close()
        break

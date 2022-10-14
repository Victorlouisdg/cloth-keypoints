import numpy as np
from cloth_manipulation.detector import get_wandb_model
from cloth_manipulation.manual_keypoints import ClothTransform
from keypoint_detection.utils.visualization import overlay_image_with_heatmap
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap
import torch
import cv2
from camera_toolkit.zed2i import Zed2i


class KeypointObserver():
    def __init__(self, wandb_checkout_reference):
        self.keypoint_detector = get_wandb_model(wandb_checkout_reference)

    def observe(self, image):
        transformed_image = ClothTransform.transform_image(image)
        image_batched = torch.Tensor(transformed_image).unsqueeze(0) / 255.0
        with torch.no_grad():
            heatmap = self.keypoint_detector(image_batched)

        heatmap_channel_batched = heatmap.squeeze(1)
        heatmap_channel = heatmap_channel_batched.squeeze(0)
        keypoints = get_keypoints_from_heatmap(
            heatmap_channel.cpu(), min_keypoint_pixel_distance=4, max_keypoints=4
        )

        # Save for visualization
        self.keypoints = keypoints
        self.transformed_image = transformed_image
        self.image_batched = image_batched
        self.heatmap_channel_batched = heatmap_channel_batched

        return keypoints

    def visualize_last_observation(self, show_heatmap=True) -> np.ndarray:
        if show_heatmap:
            overlayed = overlay_image_with_heatmap(self.image_batched, self.heatmap_channel_batched)
            overlayed = overlayed.squeeze(0).numpy()
            image = (overlayed * 255.0).astype(np.uint8)
        else:
            image = self.transformed_image.copy()

        image = Zed2i.image_shape_torch_to_opencv(image)
        image = image.copy()

        for keypoint in self.keypoints:
            image = cv2.circle(image, keypoint, 6, (0, 255, 0))
        return image
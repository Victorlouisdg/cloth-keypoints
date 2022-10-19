import cv2
import numpy as np
import torch
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.detector import get_wandb_model
from cloth_manipulation.input_transform import InputTransform
from keypoint_detection.utils.heatmap import get_keypoints_from_heatmap
from keypoint_detection.utils.visualization import overlay_image_with_heatmap


class KeypointObserver:
    def __init__(self):
        self.keypoint_detector = get_wandb_model().cuda()

    def observe(self, image):
        transformed_image = InputTransform.transform_image(image)
        image_batched = torch.Tensor(transformed_image).unsqueeze(0) / 255.0
        with torch.no_grad():
            heatmap = self.keypoint_detector(image_batched.cuda()).cpu()

        heatmap_channel_batched = heatmap.squeeze(1)
        heatmap_channel = heatmap_channel_batched.squeeze(0)
        keypoints = get_keypoints_from_heatmap(heatmap_channel.cpu(), min_keypoint_pixel_distance=4, max_keypoints=4)

        # Save for visualization
        self.keypoints = keypoints
        self.transformed_image = transformed_image
        self.image_batched = image_batched
        self.heatmap_channel_batched = heatmap_channel_batched

        return keypoints

    def visualize_last_observation(self, show_heatmap=True, competition_format=False) -> np.ndarray:
        if competition_format:
            show_heatmap = False

        if len(self.keypoints) == 0:
            return self.transformed_image.copy()

        if show_heatmap:
            overlayed = overlay_image_with_heatmap(self.image_batched, self.heatmap_channel_batched)
            overlayed = overlayed.squeeze(0).numpy()
            image = (overlayed * 255.0).astype(np.uint8)
        else:
            image = self.transformed_image.copy()

        image = Zed2i.image_shape_torch_to_opencv(image)
        image = image.copy()

        if len(self.keypoints) >= 2:
            center = np.mean(np.array(self.keypoints, float), axis=0)
        else:
            # fallback to image center if not enough keypoints
            h, w, _ = image.shape
            center = np.array([w / 2, h / 2])
        approach_points = []

        for keypoint in self.keypoints:
            if competition_format:
                center_to_keypoint = keypoint - center
                pixel_length = 25
                approach_vector = pixel_length * center_to_keypoint / np.linalg.norm(center_to_keypoint)
                approach_point = (keypoint + approach_vector).astype(int)
                approach_points.append(approach_point)
                image = cv2.line(image, keypoint, approach_point, (0, 0, 255), thickness=1)

                image = cv2.circle(image, keypoint, 1, (0, 0, 255), thickness=1)
                image = cv2.circle(image, keypoint, 7, (0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
            else:
                image = cv2.circle(image, keypoint, 1, (0, 255, 0), thickness=1)
                image = cv2.circle(image, keypoint, 10, (0, 255, 0))

        self.approach_points = approach_points
        return image

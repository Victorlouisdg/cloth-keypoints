from abc import ABC
from typing import Tuple

import numpy as np
import torch
import torchvision


class KeypointImageTransform(ABC):
    def transform_image(self, img_batch: np.ndarray):
        raise NotImplementedError

    def transform_keypoints(self, keypoint_batch: np.ndarray):
        raise NotImplementedError


class CropKeypointImageTransform(KeypointImageTransform):
    def __init__(self, start_u: int, width: int, start_v: int, heigth: int) -> None:
        super().__init__()
        self._start_u = start_u
        self._start_v = start_v
        self._width = width
        self._height = heigth

    def transform_image(self, img_batch: np.ndarray):
        assert len(img_batch.shape) == 3, "expecting shape C x H x W "
        return img_batch[
            :,
            self._start_v : self._start_v + self._height,
            self._start_u : self._start_u + self._width,
        ]

    def transform_keypoints(self, keypoint_batch: np.ndarray):
        # keypoints are now in (U,V) coordinates instead of (V,U) (HxW)
        assert np.max(keypoint_batch) > 1, "keypoints should be in absolute coordinates"
        keypoint_batch[:, 0] = keypoint_batch[:, 0] + self._start_u
        keypoint_batch[:, 1] = keypoint_batch[:, 1] + self._start_v
        return keypoint_batch


class ResizeKeypointImageTransform(KeypointImageTransform):
    def __init__(self, original_dims: Tuple[int], target_dims: Tuple[int]) -> None:
        super().__init__()
        self._original_dims = original_dims
        self._target_dims = target_dims
        self.resize_transform = torchvision.transforms.Resize(self._target_dims)

    def transform_image(self, img_batch: np.ndarray):
        img_tensor = torch.Tensor(img_batch)
        return self.resize_transform(img_tensor).numpy()

    def transform_keypoints(self, keypoint_batch: np.ndarray):
        keypoint_batch[:, 0] = keypoint_batch[:, 0] * self._original_dims[0] / self._target_dims[0]
        keypoint_batch[:, 1] = keypoint_batch[:, 1] * self._original_dims[1] / self._target_dims[1]
        return keypoint_batch


class InputTransform:
    """The transform used project-wide to transform camera images to detector inputs."""

    crop_start_u = 800
    crop_start_v = 200
    crop_height = 1000
    crop_width = 1000
    resize_height = 256
    resize_width = 256

    @classmethod
    def crop_transform(cls):
        return CropKeypointImageTransform(cls.crop_start_u, cls.crop_width, cls.crop_start_v, cls.crop_height)

    @classmethod
    def resize_transform(cls):
        return ResizeKeypointImageTransform((cls.crop_height, cls.crop_width), (cls.resize_height, cls.resize_width))

    @staticmethod
    def transform_image(img: np.ndarray):
        img = InputTransform.crop_transform().transform_image(img)
        img = InputTransform.resize_transform().transform_image(img)
        return img.astype(np.uint8)

    @staticmethod
    def reverse_transform_keypoints(keypoints: np.ndarray):
        return InputTransform.crop_transform().transform_keypoints(
            InputTransform.resize_transform().transform_keypoints(keypoints)
        )

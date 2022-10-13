import abc
import pickle
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torchvision
from camera_toolkit.zed2i import Zed2i
import cloth_manipulation.camera_mapping as cm

# load camera to marker transform
with open(Path(__file__).parent / "marker.pickle", "rb") as f:
    aruco_in_camera_position, aruco_in_camera_orientation = pickle.load(f)
# get camera extrinsics transform
aruco_in_camera_transform = np.eye(4)
aruco_in_camera_transform[:3, :3] = aruco_in_camera_orientation
aruco_in_camera_transform[:3, 3] = aruco_in_camera_position


class KeypointImageTransform(abc.ABC):
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
        keypoint_batch[:, 0] = (
            keypoint_batch[:, 0] * self._original_dims[0] / self._target_dims[0]
        )
        keypoint_batch[:, 1] = (
            keypoint_batch[:, 1] * self._original_dims[1] / self._target_dims[1]
        )
        return keypoint_batch


class ClothTransform:
    crop_start_u = 800
    crop_start_v = 200
    crop_height = 1000
    crop_width = 1000
    resize_height = 256
    resize_width = 256

    @classmethod
    def crop_transform(cls):
        return CropKeypointImageTransform(
            cls.crop_start_u, cls.crop_width, cls.crop_start_v, cls.crop_height
        )

    @classmethod
    def resize_transform(cls):
        return ResizeKeypointImageTransform(
            (cls.crop_height, cls.crop_width), (cls.resize_height, cls.resize_width)
        )

    @staticmethod
    def transform_image(img: np.ndarray):
        img = ClothTransform.crop_transform().transform_image(img)
        img = ClothTransform.resize_transform().transform_image(img)
        return img.astype(np.uint8)

    @staticmethod
    def reverse_transform_keypoints(keypoints: np.ndarray):
        return ClothTransform.crop_transform().transform_keypoints(
            ClothTransform.resize_transform().transform_keypoints(keypoints)
        )


def get_manual_keypoints(img: np.ndarray, num_keypoints: int = 4):
    """function to capture image and select some keypoints manually, which allows to test the folding w/o the state estimation"""
    # opencv mouseclick registration
    clicked_coords = []

    def clicked_callback_cv(event, x, y, flags, param):
        global u_clicked, v_clicked
        if event == cv2.EVENT_LBUTTONDBLCLK:
            print(f"clicked on {x}, {y}")
            clicked_coords.append(np.array([x, y]))

    # mark the keypoints in image plane by clicking
    cv2.imshow("img", img)
    cv2.setMouseCallback("img", clicked_callback_cv)

    while True:
        print(
            f"double click to select{num_keypoints} a keypoint; press any key after you are finished"
        )

        cv2.waitKey(0)
        if len(clicked_coords) > num_keypoints:
            raise IndexError("too many keypoint clicked, aborting.")
        elif len(clicked_coords) == num_keypoints:
            break

    cv2.destroyAllWindows()

    return clicked_coords


if __name__ == "__main__":
    from camera_toolkit.reproject import reproject_to_world_z_plane

    # open ZED (and assert it is available)
    Zed2i.list_camera_serial_numbers()
    zed = Zed2i(serial_number=cm.CameraMapping.serial_top)
    # capture image
    torch_img = zed.get_rgb_image()
    img = zed.image_shape_torch_to_opencv(torch_img)

    ## original image
    keypoints_in_camera = np.array(get_manual_keypoints(img, 4))
    print(f"{keypoints_in_camera=}")
    keypoints_in_world = reproject_to_world_z_plane(
        keypoints_in_camera, zed.get_camera_matrix(), aruco_in_camera_transform
    )
    print(keypoints_in_world)

    ### transformed

    transformed_image = ClothTransform.transform_image(torch_img)

    transformed_cv_img = Zed2i.image_shape_torch_to_opencv(transformed_image)
    transformed_keypoints = np.array(get_manual_keypoints(transformed_cv_img, 4))

    keypoints_in_camera = ClothTransform.reverse_transform_keypoints(
        transformed_keypoints
    )
    print(f"{keypoints_in_camera=}")

    keypoints_in_world = reproject_to_world_z_plane(
        keypoints_in_camera, zed.get_camera_matrix(), aruco_in_camera_transform
    )
    print(keypoints_in_world)

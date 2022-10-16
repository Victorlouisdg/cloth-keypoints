from abc import ABC, abstractmethod
from typing import List
import numpy as np
from cloth_manipulation.motion_primitives.pull import TowelReorientPull, execute_pull_primitive


class DualArmController(ABC):
    def __init__(self, dual_arm):
        self.dual_arm = dual_arm
        self.finished = False

    @abstractmethod
    def act(self, keypoints: List[np.ndarray]) -> None:
        pass

    @abstractmethod
    def visualize_plan(self, keypoints, image, world_to_camera, camera_matrix):
        pass


class ReorientTowelController(DualArmController):
    def __init__(self, dual_arm, sufficiently_low_corner_error=0.05):
        self.stopping_error = sufficiently_low_corner_error
        super().__init__(dual_arm)

    def act(self, keypoints):
        if self.finished:
            return

        if len(keypoints) != 4:
            return

        pull = TowelReorientPull(keypoints, self.dual_arm)
        if pull.average_corner_error() <= self.stopping_error:
            self.finished = True
            return

        execute_pull_primitive(pull, self.dual_arm)

    def visualize_plan(self, keypoints, image, world_to_camera, camera_matrix):
        from cloth_manipulation.gui import visualize_towel_reorient_pull  # requires cv2

        if len(keypoints) != 4:
            return image

        pull = TowelReorientPull(keypoints, self.dual_arm)
        image = visualize_towel_reorient_pull(image, pull, world_to_camera, camera_matrix)
        return image

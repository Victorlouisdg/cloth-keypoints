from abc import ABC, abstractmethod
from typing import List

import numpy as np
from camera_toolkit.reproject import project_world_to_image_plane
from cloth_manipulation.geometry import get_ordered_keypoints, move_closer
from cloth_manipulation.gui import draw_keypoints, draw_pose
from cloth_manipulation.motion_primitives.fold_execution import execute_dual_fold_trajectories
from cloth_manipulation.motion_primitives.fold_trajectory_parameterization import CircularFoldTrajectory
from cloth_manipulation.motion_primitives.pull import TowelReorientPull, execute_pull_primitive


class DualArmController(ABC):
    def __init__(self, dual_arm):
        self.dual_arm = dual_arm
        self.finished = False

    @abstractmethod
    def act(self, keypoints: List[np.ndarray]) -> None:
        pass

    @abstractmethod
    def visualize_plan(self, image, keypoints, world_to_camera, camera_matrix):
        pass


class ReorientTowelController(DualArmController):
    def __init__(self, dual_arm, sufficiently_low_corner_error=0.05):
        self.stopping_error = sufficiently_low_corner_error
        self.is_out_of_way = False
        super().__init__(dual_arm)

    def act(self, keypoints):
        if not self.is_out_of_way:
            self.dual_arm.dual_move_tcp(self.dual_arm.left.out_of_way_pose, self.dual_arm.right.out_of_way_pose)
            self.is_out_of_way = True
            return

        for keypoint in keypoints:
            assert keypoint.shape[0] == 3

        if self.finished:
            return

        if len(keypoints) != 4:
            return

        pull = TowelReorientPull(keypoints, self.dual_arm)
        if pull.average_corner_error() <= self.stopping_error:
            print(f"{__class__}: success, stopping because average corner error = {pull.average_corner_error()}")
            self.finished = True
            return

        execute_pull_primitive(pull, self.dual_arm)
        self.is_out_of_way = False

    def visualize_plan(self, image, keypoints, world_to_camera, camera_matrix):
        from cloth_manipulation.gui import visualize_towel_reorient_pull  # requires cv2

        if not self.is_out_of_way:
            return image

        image = draw_keypoints(image, keypoints, world_to_camera, camera_matrix)

        if len(keypoints) != 4:
            return image

        pull = TowelReorientPull(keypoints, self.dual_arm)
        image = visualize_towel_reorient_pull(image, pull, world_to_camera, camera_matrix)
        return image


class FoldTowelController(DualArmController):
    def __init__(self, dual_arm):
        super().__init__(dual_arm)
        self.is_out_of_way = False

    def act(self, keypoints: List[np.ndarray]) -> None:
        if not self.is_out_of_way:
            self.dual_arm.dual_move_tcp(self.dual_arm.left.out_of_way_pose, self.dual_arm.right.out_of_way_pose)
            self.is_out_of_way = True
            return

        if self.finished:
            return

        if len(keypoints) != 4:
            return

        fold_trajectory_left, fold_trajectory_right = FoldTowelController.get_fold_trajectories(keypoints)
        execute_dual_fold_trajectories(fold_trajectory_left, fold_trajectory_right, self.dual_arm)
        self.out_of_way = False
        self.finished = True

    @staticmethod
    def get_starts_and_ends(keypoints):
        corners = get_ordered_keypoints(keypoints)
        end_right, end_left, start_left, start_right = corners

        end_left, end_right = move_closer(end_left, end_right, 0.08)
        start_left, start_right = move_closer(start_left, start_right, 0.08)
        return start_left, start_right, end_left, end_right

    @staticmethod
    def get_fold_trajectories(keypoints):
        start_left, start_right, end_left, end_right = FoldTowelController.get_starts_and_ends(keypoints)
        grasp_offset = 0.025 * np.linalg.norm(start_left - start_right)
        fold_trajectory_left = CircularFoldTrajectory(start_left, end_left, -grasp_offset)
        fold_trajectory_right = CircularFoldTrajectory(start_right, end_right, grasp_offset)
        return fold_trajectory_left, fold_trajectory_right

    def visualize_plan(self, image, keypoints, world_to_camera, camera_matrix):
        import cv2

        if not self.is_out_of_way:
            return image

        image = draw_keypoints(image, keypoints, world_to_camera, camera_matrix)

        if len(keypoints) != 4:
            return

        fold_trajectory_left, fold_trajectory_right = FoldTowelController.get_fold_trajectories(keypoints)

        def draw_trajectory(image, trajectory, world_to_camera, camera_matrix):
            waypoints = [trajectory._fold_pose(completion)[:3, -1] for completion in np.linspace(0, 1, 20)]
            waypoints_image = [
                project_world_to_image_plane(waypoint, world_to_camera, camera_matrix) for waypoint in waypoints
            ]
            waypoints_image = np.array(waypoints_image, np.int32).reshape((-1, 1, 2))
            image = cv2.polylines(image, [waypoints_image], isClosed=False, color=(0, 255, 255), thickness=2)

            for completion in np.linspace(0, 1, 3):
                image = draw_pose(image, trajectory._fold_pose(completion), world_to_camera, camera_matrix)

            image = draw_pose(image, trajectory.get_pregrasp_pose(), world_to_camera, camera_matrix)
            image = draw_pose(image, trajectory.get_fold_retreat_pose(), world_to_camera, camera_matrix)
            return image

        left_center, right_center = fold_trajectory_left.center, fold_trajectory_right.center
        left_center, right_center = move_closer(left_center, right_center, -0.2)
        left_center = project_world_to_image_plane(left_center, world_to_camera, camera_matrix).astype(int)
        right_center = project_world_to_image_plane(right_center, world_to_camera, camera_matrix).astype(int)
        image = cv2.line(image, left_center, right_center, color=(255, 0, 255), thickness=2)

        image = draw_trajectory(image, fold_trajectory_left, world_to_camera, camera_matrix)
        image = draw_trajectory(image, fold_trajectory_right, world_to_camera, camera_matrix)

        return image


class ReorientAndFoldTowelController(DualArmController):
    def __init__(self, dual_arm):
        self.dual_arm = dual_arm
        self.finished = False
        self.reorient_controller = ReorientTowelController(dual_arm)
        self.fold_controller = FoldTowelController(dual_arm)
        self.second_round_started = False

    def act(self, keypoints: List[np.ndarray]) -> None:
        if self.finished:
            return

        if len(keypoints) != 4:
            return

        if not self.reorient_controller.finished:
            self.reorient_controller.act(keypoints)
            return

        if not self.fold_controller.finished:
            self.fold_controller.act(keypoints)
            return

        if not self.second_round_started:
            self.reorient_controller.finished = False
            self.fold_controller.finished = False
            self.second_round_started = True
            return

        self.finished = True

    def visualize_plan(self, image, keypoints, world_to_camera, camera_matrix):
        if not self.reorient_controller.finished:
            image = self.reorient_controller.visualize_plan(image, keypoints, world_to_camera, camera_matrix)
            return image

        if not self.fold_controller.finished:
            image = self.fold_controller.visualize_plan(image, keypoints, world_to_camera, camera_matrix)
            return image

        return image

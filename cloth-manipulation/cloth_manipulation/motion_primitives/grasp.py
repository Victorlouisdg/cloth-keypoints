from abc import ABC, abstractmethod
from typing import List

import numpy as np
from cloth_manipulation.controllers import DualArmController
from cloth_manipulation.geometry import get_ordered_keypoints, pose_from_orientation_and_position
from cloth_manipulation.gui import draw_keypoints, draw_pose
from cloth_manipulation.hardware.base_classes import DualArm
from scipy.spatial.transform import Rotation


class GraspOrthogonalTowelEdgeController(DualArmController):
    def __init__(self, dual_arm):
        self.dual_arm = dual_arm
        self.finished = False
        self.is_out_of_way = False

    def get_grasp(self, keypoints):
        corners = get_ordered_keypoints(keypoints)
        grasp = GraspOrthogonalTowelEdge(corners, (0, 1), 0.05)
        return grasp

    def act(self, keypoints: List[np.ndarray]) -> None:
        if not self.is_out_of_way:
            self.dual_arm.dual_move_tcp(self.dual_arm.left.out_of_way_pose, self.dual_arm.right.out_of_way_pose)
            self.is_out_of_way = True
            return

        if self.finished:
            return

        if len(keypoints) != 4:
            return

        grasp = self.get_grasp(keypoints)
        execute_grasp(grasp, self.dual_arm)
        self.out_of_way = False
        self.finished = True

    def visualize_plan(self, image, keypoints, world_to_camera, camera_matrix):
        image = draw_keypoints(image, keypoints, world_to_camera, camera_matrix)

        corners = get_ordered_keypoints(keypoints)
        grasp = GraspOrthogonalTowelEdge(corners, (0, 1), 0.05)

        image = draw_pose(image, grasp.get_grasp_pose(), world_to_camera, camera_matrix)
        image = draw_pose(image, grasp.get_pregrasp_pose(), world_to_camera, camera_matrix)

        return image


class Grasp(ABC):
    @abstractmethod
    def get_grasp_pose(self):
        pass

    @abstractmethod
    def get_pregrasp_pose(self, offset=0.05):
        pass


class GraspOrthogonalTowelEdge(Grasp):
    def __init__(self, corners, edge, edge_distance, angle_with_table=45.0, grasp_depth=0.05):
        self.grasp_depth = grasp_depth
        v0, v1 = edge
        corner0, corner1 = corners[v0], corners[v1]
        edge_vector = corner1 - corner0

        edge_unit = edge_vector / np.linalg.norm(edge_vector)

        # construct pose parallel to table
        edge_grasp_location = corner0 + edge_distance * edge_unit

        gripper_y = edge_unit
        gripper_x = np.array([0, 0, 1])
        gripper_z = np.cross(gripper_x, gripper_y)

        flat_orientation = np.column_stack([gripper_x, gripper_y, gripper_z])

        # grasp_pose_flat = transformation_matrix_from_position_and_vecs(grasp_location, gripper_x, gripper_y, gripper_z)

        # self.grasp_pose = grasp_pose_flat
        self.approach_direction = gripper_z.copy()
        grasp_location = edge_grasp_location + grasp_depth * self.approach_direction

        rotation = Rotation.from_rotvec(-np.deg2rad(angle_with_table) * gripper_y)
        tilted_orientation = rotation.as_matrix() @ flat_orientation

        self.grasp_pose = pose_from_orientation_and_position(tilted_orientation, grasp_location)

    def get_grasp_pose(self):
        return self.grasp_pose.copy()

    def get_pregrasp_pose(self, offset=0.05):
        pregrasp_pose = self.get_grasp_pose()
        pregrasp_pose[:3, -1] -= (self.grasp_depth + offset) * self.approach_direction
        return pregrasp_pose


def execute_grasp(grasp: Grasp, dual_arm: DualArm):
    dual_arm.dual_move_tcp(dual_arm.left.home_pose, dual_arm.right.home_pose)
    robot = dual_arm.left
    robot.gripper.close()
    robot.move_tcp(grasp.get_pregrasp_pose())
    robot.gripper.open()
    robot.move_tcp_linear(grasp.get_grasp_pose(), speed=robot.LINEAR_SPEED, acceleration=robot.LINEAR_ACCELERATION)
    robot.gripper.close()

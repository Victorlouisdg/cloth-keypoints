from typing import List

import numpy as np
from cloth_manipulation.ur_robotiq_dual_arm_interface import DualArmUR, homogeneous_pose_to_position_and_rotvec
from cloth_manipulation.utils import get_ordered_keypoints


class PullPrimitive:
    def __init__(self, start: np.ndarray, end: np.ndarray) -> None:
        self.start_position = start
        self.end_position = end

        # top down gripper orientation
        self.gripper_orientation = np.eye(3)
        self.gripper_orientation[2, 2] = -1
        self.gripper_orientation[0, 0] = -1

    def get_pre_grasp_pose(self):
        pregrasp_pose = self.get_pull_start_pose()
        pregrasp_pose[2, 3] += 0.05
        return pregrasp_pose

    def get_pull_start_pose(self):
        start_pose = np.eye(4)
        start_pose[:3, :3] = self.gripper_orientation
        start_pose[:3, 3] = self.start_position
        return start_pose

    def get_pull_end_pose(self):
        end_pose = np.eye(4)
        end_pose[:3, :3] = self.gripper_orientation
        end_pose[:3, 3] = self.end_position
        return end_pose

    def get_pull_retreat_pose(self):
        retreat_pose = self.get_pull_end_pose()
        retreat_pose[2, 3] += 0.05
        return retreat_pose

    def __repr__(self) -> str:
        return f"{self.start_position=} -> {self.end_position=}"


def select_towel_pull(corners: List[np.ndarray], margin=0.05) -> PullPrimitive:
    def vector_cosine(v0, v1):
        return np.dot(v0, v1) / np.linalg.norm(v0) / np.linalg.norm(v1)

    def closest_point(point, candidates):
        distances = [np.linalg.norm(point - candidate) for candidate in candidates]
        return candidates[np.argmin(distances)]

    corners = np.array(corners)
    towel_center = np.mean(corners, axis=0)

    corners = get_ordered_keypoints(corners)
    edges = [(i, (i + 1) % 4) for i in range(4)]

    edge_lengths = [np.linalg.norm(corners[id0] - corners[id1]) for (id0, id1) in edges]

    edge_pairs = [(0, 2), (1, 3)]
    edge_pairs_mean_length = []
    for eid0, eid1 in edge_pairs:
        edge_length_mean = np.mean([edge_lengths[eid0], edge_lengths[eid1]])
        edge_pairs_mean_length.append(edge_length_mean)

    short_edge_pair = edge_pairs[np.argmin(edge_pairs_mean_length)]
    short_edges = [edges[eid] for eid in short_edge_pair]

    # By convention widht is smaller than length
    towel_width = min(edge_pairs_mean_length)
    towel_length = max(edge_pairs_mean_length)

    # We want the short edges parallel to the x-axis
    tx = towel_width / 2
    ty = towel_length / 2
    desired_corners = [
        np.array([tx, ty, 0]),
        np.array([-tx, ty, 0]),
        np.array([-tx, -ty, 0]),
        np.array([tx, -ty, 0]),
    ]

    corner_destination_pairs = []
    for edge in short_edges:
        id0, id1 = edge
        corner0 = corners[id0]
        corner0_options = [desired_corners[0], desired_corners[2]]
        destination0 = closest_point(corner0, corner0_options)

        corner1 = corners[id1]
        corner1_options = [desired_corners[1], desired_corners[3]]
        destination1 = closest_point(corner1, corner1_options)

        corner_destination_pairs.append((corner0, destination0))
        corner_destination_pairs.append((corner1, destination1))

    cosines = []
    for corner, destination in corner_destination_pairs:
        center_to_corner = corner - towel_center
        corner_to_destination = destination - corner
        cosine = vector_cosine(center_to_corner, corner_to_destination)
        cosines.append(cosine)

    best_pair = corner_destination_pairs[np.argmax(cosines)]
    corner, destination = best_pair

    corner_to_center = towel_center - corner
    corner_to_center_unit = corner_to_center / np.linalg.norm(corner_to_center)
    margin_vector = corner_to_center_unit * margin

    start = corner + margin_vector
    end = destination + margin_vector
    return PullPrimitive(start, end)


def execute_pull_primitive(pull_primitive: PullPrimitive, dual_arm: DualArmUR):

    # decide which robot to use
    reachable_by_victor = (
        dual_arm.victor_ur.is_world_pose_reachable(
            homogeneous_pose_to_position_and_rotvec(pull_primitive.get_pull_start_pose())
        )
        and dual_arm.victor_ur.is_world_pose_reachable(
            homogeneous_pose_to_position_and_rotvec(pull_primitive.get_pull_end_pose())
        )
        and dual_arm.victor_ur.is_world_pose_reachable(
            homogeneous_pose_to_position_and_rotvec(pull_primitive.get_pull_retreat_pose())
        )
    )
    if reachable_by_victor:
        ur = dual_arm.victor_ur
    else:
        ur = dual_arm.louise_ur

    ur.gripper.gripper.move_to_position(200)  # little bit more compliant if finger tips don't touch
    # go to home pose
    ur.moveL(ur.home_pose, vel=2 * ur.DEFAULT_LINEAR_VEL)
    # go to prepull pose
    ur.moveL(
        homogeneous_pose_to_position_and_rotvec(pull_primitive.get_pre_grasp_pose()), vel=2 * ur.DEFAULT_LINEAR_VEL
    )
    # move down
    ur.moveL(homogeneous_pose_to_position_and_rotvec(pull_primitive.get_pull_start_pose()))

    # pull
    ur.moveL(homogeneous_pose_to_position_and_rotvec(pull_primitive.get_pull_end_pose()))

    # move up
    ur.moveL(homogeneous_pose_to_position_and_rotvec(pull_primitive.get_pull_retreat_pose()))

    # move to home pose
    ur.moveL(ur.home_pose, vel=2 * ur.DEFAULT_LINEAR_VEL)

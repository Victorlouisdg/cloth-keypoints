import numpy as np
from cloth_manipulation.geometry import angle_2D, get_ordered_keypoints, get_short_and_long_edges, rotate_point
from cloth_manipulation.hardware.base_classes import DualArm
from cloth_manipulation.motion_primitives.pull import PullPrimitive
from scipy.spatial.transform import Rotation


class PickReorientTowelPull(PullPrimitive):
    def __init__(self, corners, dual_arm: DualArm, inset_amount=0.05, compliance_distance=0.002):
        self.dual_arm = dual_arm
        self.corners = corners
        self.select_towel_pulls(corners)
        self.set_orientations()
        # self.start, self.end = self.inset_pull_positions(inset_amount)
        # self.start[2] -= compliance_distance
        # self.end[2] -= compliance_distance
        # super().__init__(self.start, self.end)
        # self.set_robot_and_orientations(self.start, self.end, dual_arm)

    @staticmethod
    def vector_cosine(v0, v1):
        return np.dot(v0, v1) / np.linalg.norm(v0) / np.linalg.norm(v1)

    @staticmethod
    def closest_point(point, candidates):
        distances = [np.linalg.norm(point - candidate) for candidate in candidates]
        return candidates[np.argmin(distances)]

    @staticmethod
    def top_down_orientation(gripper_open_direction):
        X = gripper_open_direction / np.linalg.norm(gripper_open_direction)  # np.array([-1, 0, 0])
        Z = np.array([0, 0, -1])
        Y = np.cross(Z, X)
        return np.column_stack([X, Y, Z])

    @staticmethod
    def tilted_pull_orientation(pull_location, robot_location, tilt_angle=15):
        robot_to_pull = pull_location - robot_location
        if np.linalg.norm(robot_to_pull) < 0.35:
            tilt_angle = -tilt_angle  # tilt inwards
        gripper_open_direction = robot_to_pull
        top_down = PickReorientTowelPull.top_down_orientation(gripper_open_direction)

        gripper_y = top_down[:, 1]
        rotation = Rotation.from_rotvec(np.deg2rad(tilt_angle) * gripper_y)

        gripper_orienation = rotation.as_matrix() @ top_down

        return gripper_orienation

    @staticmethod
    def get_desired_corners(ordered_corners):
        corners = ordered_corners
        short_edges, _ = get_short_and_long_edges(corners)
        middles = []
        for edge in short_edges:
            corner0 = corners[edge[0]]
            corner1 = corners[edge[1]]
            middle = (corner0 + corner1) / 2
            middles.append(middle)

        # Ensure the middle with highest y-value is first
        if middles[0][1] < middles[1][1]:
            middles.reverse()

        towel_y_axis = middles[0] - middles[1]
        y_axis = [0, 1]

        angle = angle_2D(towel_y_axis, y_axis)
        towel_center = np.mean(corners, axis=0)
        z_axis = np.array([0, 0, 1])

        rotated_corners = [rotate_point(corner, towel_center, z_axis, angle) for corner in corners]
        desired_corners = [corner - towel_center for corner in rotated_corners]
        return desired_corners

    @staticmethod
    def select_best_pull_ids(corners, desired_corners):
        towel_center = np.mean(corners, axis=0)
        scores = []
        for corner, desired in zip(corners, desired_corners):
            center_to_corner = corner - towel_center
            pull = desired - corner
            if np.linalg.norm(pull) < 0.05:
                scores.append(-1)
                continue
            alignment = PickReorientTowelPull.vector_cosine(center_to_corner, pull)
            scores.append(alignment)

        return np.argsort(scores)[
            -2:
        ]  # np.array(corners)[np.argsort(scores)][-2:], np.array(desired_corners)[np.argsort(scores)][-2:]

        # end = desired_corners[np.argmax(scores)]
        # return start, end

    def select_towel_pulls(self, corners):
        corners = np.array(corners)
        corners = get_ordered_keypoints(corners)
        self.ordered_corners = corners

        desired_corners = PickReorientTowelPull.get_desired_corners(corners)
        self.desired_corners = desired_corners

        ids = PickReorientTowelPull.select_best_pull_ids(corners, desired_corners)

        start_positions = corners[ids]
        end_positions = desired_corners[ids]

        self.start_positions = start_positions
        self.end_positions = end_positions

        left_base = self.dual_arm.left.robot_in_world_pose[:3, -1]
        right_base = self.dual_arm.right.robot_in_world_pose[:3, -1]

        left_distances = [np.linalg.norm(start - left_base) for start in start_positions]
        right_distances = [np.linalg.norm(start - right_base) for start in start_positions]

        left_id = ids[np.argmin(left_distances)]
        right_id = ids[np.argmin(right_distances)]

        if left_id != right_id:
            self.left_id = left_id
            self.right_id = right_id
            return

        id = left_id
        if left_distances[id] > right_distances[id]:
            self.left_id = id
            self.right_id = ids[np.argmax(right_distances)]
        else:
            self.left_id = ids[np.argmax(left_distances)]
            self.right_id = id

    def set_orientations(self):
        left_start = self.left_robot_start
        right_start = self.right_robot_start

        left_start - right_start

        # top_down_orientation

        # return start, end

    def inset_pull_positions(self, margin=0.05):
        """Moves the start and end positions toward the center of the towel.
        This can increase robustness to keypoint detection inaccuracy."""
        corners = np.array(self.corners)
        towel_center = np.mean(corners, axis=0)
        start_to_center = towel_center - self.start_original
        start_to_center_unit = start_to_center / np.linalg.norm(start_to_center)
        start_margin_vector = start_to_center_unit * margin
        start = self.start_original + start_margin_vector

        desired_corners = np.array(self.desired_corners)
        desired_towel_center = np.mean(desired_corners, axis=0)
        end_to_center = desired_towel_center - self.end_original
        end_to_center_unit = end_to_center / np.linalg.norm(end_to_center)
        end_margin_vector = end_to_center_unit * margin
        end = self.end_original + end_margin_vector
        return start, end

    def average_corner_error(self):
        return np.mean(
            [np.linalg.norm(corner - desired) for corner, desired in zip(self.ordered_corners, self.desired_corners)]
        )

    def set_robot_and_orientations(self, start, end, dual_arm: DualArm):
        for robot in dual_arm.arms:
            start_orientation = self.tilted_pull_orientation(start, robot.robot_in_world_pose[:3, -1])
            end_orientation = self.tilted_pull_orientation(end, robot.robot_in_world_pose[:3, -1])

            start_pose = np.eye(4)
            start_pose[:3, :3] = start_orientation
            start_pose[:3, 3] = start
            end_pose = np.eye(4)
            end_pose[:3, :3] = end_orientation
            end_pose[:3, 3] = end

            if not robot.is_pose_unsafe(start_pose) and not robot.is_pose_unsafe(end_pose):
                self.start_pose, self.end_pose, self.robot = start_pose, end_pose, robot
                return

        raise ValueError(f"Pull could not be executed by either robot. \nStart: \n{start} \nEnd: \n{end}")


def execute_pick_pull_primitive(pull_primitive: PickReorientTowelPull, dual_arm: DualArm):
    # Decide which robot to use. The ReorientTowelPull already chooses this itself.

    dual_arm.dual_gripper_move_to_position(0.8)

    # robot.gripper.move_to_position(0.8)  # little bit more compliant if finger tips don't touch
    # # go to home pose
    # robot.move_tcp(robot.home_pose)
    # # go to prepull pose
    # robot.move_tcp(pull_primitive.get_pre_grasp_pose())
    # # move down in a straight line
    # robot.move_tcp_linear(
    #     pull_primitive.get_pull_start_pose(), speed=robot.LINEAR_SPEED, acceleration=robot.LINEAR_ACCELERATION
    # )
    # # pull in a straight
    # robot.move_tcp_linear(
    #     pull_primitive.get_pull_end_pose(), speed=robot.LINEAR_SPEED, acceleration=robot.LINEAR_ACCELERATION
    # )

    # # move straight up and away
    # robot.move_tcp_linear(
    #     pull_primitive.get_pull_retreat_pose(), speed=robot.LINEAR_SPEED, acceleration=robot.LINEAR_ACCELERATION
    # )
    # robot.gripper.open()

    # # move to home pose
    # robot.move_tcp(robot.home_pose)

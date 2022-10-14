import numpy as np
from cloth_manipulation.ur_robotiq_dual_arm_interface import DualArmUR, UR, homogeneous_pose_to_position_and_rotvec
from cloth_manipulation.utils import angle_2D, get_ordered_keypoints, get_short_and_long_edges, rotate_point
from scipy.spatial.transform import Rotation


class PullPrimitive:
    def __init__(self, start: np.ndarray, end: np.ndarray, robot: UR = None) -> None:
        self.robot = robot 
        self.start_pose = self.get_topdown_pose_from_position(start)
        self.end_pose = self.get_topdown_pose_from_position(end)

    @staticmethod
    def get_topdown_pose_from_position(position):
        # top down gripper orientation
        gripper_orientation = np.eye(3)
        gripper_orientation[2, 2] = -1
        gripper_orientation[0, 0] = -1
        pose = np.eye(4)
        pose[:3, :3] = gripper_orientation
        pose[:3, 3] = position
        return pose


    def get_pre_grasp_pose(self):
        pregrasp_pose = np.copy(self.get_pull_start_pose())
        pregrasp_pose[2, 3] += 0.05
        return pregrasp_pose

    def get_pull_start_pose(self):
        return self.start_pose

    def get_pull_end_pose(self):
        return self.end_pose

    def get_pull_retreat_pose(self):
        retreat_pose = np.copy(self.get_pull_end_pose())
        retreat_pose[2, 3] += 0.05
        return retreat_pose

    def __repr__(self) -> str:
        return f"pull {self.start_pose[:3,3]=} -> {self.end_pose[:3,3]=}"


class TowelReorientPull(PullPrimitive):
    def __init__(self, corners,dual_arm: DualArmUR, inset_amount=0.05):
        self.corners = corners
        self.start_original, self.end_original = self.select_towel_pull(corners)
        self.start, self.end = self.inset_pull_positions(inset_amount)
        super().__init__(self.start, self.end)
        self.set_robot_and_orientations(self.start,self.end,dual_arm)

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
            tilt_angle = -tilt_angle # tilt inwards
        gripper_open_direction = robot_to_pull
        top_down = TowelReorientPull.top_down_orientation(gripper_open_direction)

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
    def select_best_pull_positions(corners, desired_corners):
        towel_center = np.mean(corners, axis=0)
        scores = []
        for corner, desired in zip(corners, desired_corners):
            center_to_corner = corner - towel_center
            pull = desired - corner
            if np.linalg.norm(pull) < 0.05:
                scores.append(-1)
                continue
            alignment = TowelReorientPull.vector_cosine(center_to_corner, pull)
            scores.append(alignment)

        start = corners[np.argmax(scores)]
        end = desired_corners[np.argmax(scores)]
        return start, end

    def select_towel_pull(self, corners):
        corners = np.array(corners)
        corners = get_ordered_keypoints(corners)
        self.ordered_corners = corners

        desired_corners = TowelReorientPull.get_desired_corners(corners)
        self.desired_corners = desired_corners

        start, end = TowelReorientPull.select_best_pull_positions(corners, desired_corners)
        return start, end

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

    def set_robot_and_orientations(self,start,end, dual_arm: DualArmUR):
        for robot in [dual_arm.victor_ur, dual_arm.louise_ur]:
            start_orientation = self.tilted_pull_orientation(start,robot.robot_in_world_position)
            end_orientation = self.tilted_pull_orientation(end,robot.robot_in_world_position)

            start_pose = np.eye(4)
            start_pose[:3,:3] = start_orientation
            start_pose[:3,3] = start
            end_pose = np.eye(4)
            end_pose[:3,:3] = end_orientation
            end_pose[:3,3] = end

            if robot.is_world_pose_reachable(homogeneous_pose_to_position_and_rotvec(start_pose)) and robot.is_world_pose_reachable(homogeneous_pose_to_position_and_rotvec(end_pose)):
                self.start_pose,self.end_pose, self.robot = start_pose, end_pose, robot
                return
        
        raise ValueError(f" pull with start = {start} and end={end} could not be executed by either robot")


def execute_pull_primitive(pull_primitive: PullPrimitive, dual_arm: DualArmUR):

    # decide which robot to use
    if isinstance(pull_primitive, TowelReorientPull):
        ur = pull_primitive.robot
    else:
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
    ur.gripper.gripper.open()
    # move to home pose
    ur.moveL(ur.home_pose, vel=2 * ur.DEFAULT_LINEAR_VEL)

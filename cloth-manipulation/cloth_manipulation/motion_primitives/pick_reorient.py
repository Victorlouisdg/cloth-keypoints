import numpy as np
from cloth_manipulation.controllers import DualArmController
from cloth_manipulation.geometry import get_ordered_keypoints, pose_from_orientation_and_position
from cloth_manipulation.gui import draw_keypoints
from cloth_manipulation.hardware.base_classes import DualArm
from cloth_manipulation.motion_primitives.grasp import GraspTowelPoint
from cloth_manipulation.motion_primitives.pull import PullPrimitive, ReorientTowelPull


class PickReorientTowelPull(PullPrimitive):
    def __init__(self, corners, dual_arm: DualArm):
        self.dual_arm = dual_arm
        self.corners = corners

        corners = np.array(corners)
        corners = get_ordered_keypoints(corners)
        self.ordered_corners = corners

        self.desired_corners, _ = ReorientTowelPull.get_desired_corners(corners)
        start, end, corner_id = ReorientTowelPull.select_best_pull_positions(corners, self.desired_corners)
        self.corner_id = corner_id
        self.start = start
        self.end = end

    def average_corner_error(self):
        return np.mean(
            [np.linalg.norm(corner - desired) for corner, desired in zip(self.ordered_corners, self.desired_corners)]
        )

    @property
    def direction(self):
        start_to_end = self.end - self.start
        return start_to_end / np.linalg.norm(start_to_end)


class PickReorientTowelController(DualArmController):
    def __init__(self, dual_arm, sufficiently_low_corner_error=0.05):
        self.stopping_error = sufficiently_low_corner_error
        self.is_out_of_way = False
        super().__init__(dual_arm)

    def get_plan(self, keypoints):
        pull = PickReorientTowelPull(keypoints, self.dual_arm)

        start_id = pull.corner_id
        opposite_id = (start_id + 2) % 4
        opposite_corner = pull.ordered_corners[opposite_id]
        grasp_direction = opposite_corner - pull.start
        grasp_direction /= np.linalg.norm(grasp_direction)

        grasp_depth = 0.05

        grasp = GraspTowelPoint(pull.start, grasp_direction, angle_with_table=90, grasp_depth=grasp_depth)

        left_base = self.dual_arm.left.robot_in_world_pose[:3, -1]
        right_base = self.dual_arm.right.robot_in_world_pose[:3, -1]

        left_distance = np.linalg.norm(left_base - pull.start)
        right_distance = np.linalg.norm(right_base - pull.start)

        end_diagonal = pull.desired_corners[opposite_id] - pull.desired_corners[start_id]

        if left_distance <= right_distance:
            robot = self.dual_arm.left
            robot_out_of_way = self.dual_arm.right
        else:
            robot = self.dual_arm.right
            robot_out_of_way = self.dual_arm.left

        base = robot.robot_in_world_pose[:3, -1]
        end_diagonal_unit = end_diagonal / np.linalg.norm(end_diagonal)
        end_location = pull.end + grasp_depth * end_diagonal_unit
        orientation = ReorientTowelPull.tilted_pull_orientation(end_location, base)
        pull_end_pose = pose_from_orientation_and_position(orientation, end_location)

        return grasp, pull, pull_end_pose, robot, robot_out_of_way

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

        grasp, pull, pull_end_pose, robot, robot_out_of_way = self.get_plan(keypoints)

        if pull.average_corner_error() <= self.stopping_error:
            print(f"{__class__}: success, stopping because average corner error = {pull.average_corner_error()}")
            self.finished = True
            return

        print(f"Using {robot.name}")
        execute_pick_and_reorient(grasp, pull_end_pose, robot, robot_out_of_way)
        self.is_out_of_way = False

    def visualize_plan(self, image, keypoints, world_to_camera, camera_matrix):
        from cloth_manipulation.gui import visualize_pick_reorient_towel_pull  # requires cv2

        if not self.is_out_of_way:
            return image

        image = draw_keypoints(image, keypoints, world_to_camera, camera_matrix)

        if len(keypoints) != 4:
            return image

        grasp, pull, pull_end_pose, _, _ = self.get_plan(keypoints)
        image = visualize_pick_reorient_towel_pull(image, grasp, pull, pull_end_pose, world_to_camera, camera_matrix)
        return image


def execute_pick_and_reorient(grasp, pull_end_pose, robot, robot_out_of_way):
    # robot.move_tcp(robot.home_pose)
    # robot.move_tcp(robot.home_pose)

    temp_dual = DualArm(robot, robot_out_of_way)
    temp_dual.dual_move_tcp(robot.home_pose, robot_out_of_way.out_of_way_pose)

    robot.gripper.close()
    # robot.move_tcp(grasp.get_pregrasp_pose())

    robot.move_tcp_linear(grasp.get_grasp_pose(), speed=robot.LINEAR_SPEED, acceleration=robot.LINEAR_ACCELERATION)

    robot.gripper.open()

    robot.gripper.close()

    robot.move_tcp_linear(pull_end_pose, speed=robot.LINEAR_SPEED, acceleration=robot.LINEAR_ACCELERATION)

    robot.gripper.open()
    robot.move_tcp(robot.home_pose)


# def execute_pick_pull_primitive(pull_primitive: PickReorientTowelPull, dual_arm: DualArm):
#     # Decide which robot to use. The ReorientTowelPull already chooses this itself.

#     dual_arm.dual_gripper_move_to_position(0.8)

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

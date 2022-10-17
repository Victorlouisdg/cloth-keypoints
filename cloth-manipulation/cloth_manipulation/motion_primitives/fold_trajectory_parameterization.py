from typing import List

import numpy as np
from cloth_manipulation.geometry import top_down_orientation


def transformation_matrix_from_position_and_vecs(pos, x, y, z):
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, 0] = x
    transformation_matrix[:3, 1] = y
    transformation_matrix[:3, 2] = z
    transformation_matrix[:3, 3] = pos
    return transformation_matrix


class FoldTrajectory:
    def __init__(self, start: np.ndarray, end: np.ndarray) -> None:

        # create local Frame
        self.center = (start + end) / 2
        self.x = end - start
        self.len = np.linalg.norm(self.x)
        self.x /= self.len
        self.z = np.array([0, 0, 1])

        self.y = np.cross(self.z, self.x)
        self.fold_frame_in_robot_frame = transformation_matrix_from_position_and_vecs(
            self.center, self.x, self.y, self.z
        )

    def _fold_pose(self, t):
        raise NotImplementedError

    def get_grasp_pose(self):
        return self._fold_pose(0)

    def get_pregrasp_pose(self, offest=0.05):
        raise NotImplementedError

    def get_retreat_pose(self, offest=0.05):
        raise NotImplementedError

    def get_fold_path(self, n_waypoints: int = 50) -> List[np.ndarray]:
        """Samples n_waypoints from the fold path and return them as a list of 4x4 poses."""
        waypoints = [self._fold_pose(completion) for completion in np.linspace(0, 1, n_waypoints)]
        return np.array(waypoints)


class CircularFoldTrajectory(FoldTrajectory):
    def __init__(self, start, end) -> None:
        super().__init__(start, end)

    def _fold_pose(self, t) -> np.ndarray:
        """Parameterization of the fold trajectory
        t = 0 is the grasp pose, t = 1 is the final (release) pose
        """
        assert t <= 1 and t >= 0
        position_angle = np.pi - t * np.pi
        # the radius was manually tuned on a cloth to find a balance between grasp width along the cloth and grasp robustness given the gripper fingers.
        radius = self.len / 2.0 - 0.04
        position = np.array([radius * np.cos(position_angle), 0, radius * np.sin(position_angle)])

        grasp_angle = np.pi / 10
        # bring finger tip down to zero.
        position[2] += (0.085 / 2 * np.sin(grasp_angle) - 0.008) * np.cos(
            grasp_angle
        )  # want the low finger to touch the table so offset from TCP
        position[2] -= 0.008  # 8mm compliance for better grasping

        # orientation_angle = max(grasp_angle - t * 2 * grasp_angle, -np.pi / 4)
        orientation_angle = (t * -np.pi / 4) + (1 - t) * grasp_angle
        x = np.array([np.cos(orientation_angle), 0, np.sin(orientation_angle)])
        x /= np.linalg.norm(x)
        y = np.array([0, -1, 0])

        z = np.cross(x, y)
        return self.fold_frame_in_robot_frame @ transformation_matrix_from_position_and_vecs(position, x, y, z)

    def get_pregrasp_pose(self, offset=0.05):
        grasp_pose = self._fold_pose(0)
        end_pose = self._fold_pose(1)
        pregrasp_pose = grasp_pose

        start = grasp_pose[:3, -1]
        end = end_pose[:3, -1]
        start_to_end = end - start
        start_to_end /= np.linalg.norm(start_to_end)

        pregrasp_pose[:3, -1] += -offset * start_to_end
        return pregrasp_pose

    def get_fold_retreat_pose(self, offset=0.05):
        self._fold_pose(0)
        end_pose = self._fold_pose(1)

        retreat_pose = end_pose
        gripper_forward = end_pose[:3, 2]  # gripper Z-axis
        gripper_open_direction = np.array([0, 1, 0])  # gripper X-axis

        # translate end pose away form gripper forward dir
        retreat_pose[:3, -1] += -offset * gripper_forward
        retreat_pose[:3, :3] = top_down_orientation(gripper_open_direction)
        return retreat_pose


class VLFoldLine(FoldTrajectory):
    def __init__(self, start: np.ndarray, end: np.ndarray) -> None:
        super().__init__(start, end)
        raise NotImplementedError
        # TODO: allow for time parameterization (through linear velocity at each waypoint) in the trajectory generation.


if __name__ == "__main__":
    s = CircularFoldTrajectory(np.array([0.20, -0.2, 0.2]), np.array([0.2, 0.2, 0.2]))
    print(s.get_pregrasp_pose())
    print(s.get_grasp_pose())
    print(s.get_fold_path(4))
    print(s.get_fold_retreat_pose())

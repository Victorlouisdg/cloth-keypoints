import numpy as np
from scipy.spatial.transform import Rotation


def angle_2D(v0, v1):
    # TODO: document.
    x1, y1, *_ = v0
    x2, y2, *_ = v1
    dot = x1 * x2 + y1 * y2  # dot product between [x1, y1] and [x2, y2]
    det = x1 * y2 - y1 * x2  # determinant
    angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    return angle


def rotate_point(point, rotation_origin, rotation_axis, angle):
    unit_axis = rotation_axis / np.linalg.norm(rotation_axis)
    rotation = Rotation.from_rotvec(angle * unit_axis)
    point_new = rotation.as_matrix() @ (point - rotation_origin) + rotation_origin
    return point_new


def get_ordered_keypoints(keypoints):
    """
    orders keypoints according to their angle w.r.t. a frame that is created by translating the world frame to the center of the cloth.
    the first keypoints is the one with the smallest, positive angle and they are sorted counter-clockwise.
    """
    keypoints = np.array(keypoints)
    center = np.mean(keypoints, axis=0)
    x_axis = np.array([1, 0])
    angles = [angle_2D(x_axis, keypoint - center) for keypoint in keypoints]
    angles = [angle % (2 * np.pi) for angle in angles]  # make angles positive from 0 to 2*pi
    keypoints_sorted = keypoints[np.argsort(angles)]
    return list(keypoints_sorted)


def get_short_and_long_edges(ordered_corners):
    edges = [(i, (i + 1) % 4) for i in range(4)]
    edge_lengths = [np.linalg.norm(ordered_corners[id0] - ordered_corners[id1]) for (id0, id1) in edges]
    edge_pairs = [(0, 2), (1, 3)]
    edge_pairs_mean_length = []
    for eid0, eid1 in edge_pairs:
        edge_length_mean = np.mean([edge_lengths[eid0], edge_lengths[eid1]])
        edge_pairs_mean_length.append(edge_length_mean)

    short_edge_pair = edge_pairs[np.argmin(edge_pairs_mean_length)]
    short_edges = [edges[eid] for eid in short_edge_pair]

    long_edge_pair = edge_pairs[np.argmin(edge_pairs_mean_length)]
    long_edges = [edges[eid] for eid in long_edge_pair]
    return short_edges, long_edges


def move_closer(point0: np.ndarray, point1: np.ndarray, distance: float):
    """Moves two points closer to each other by a distance. Each point will have moved half of this distance."""
    direction = point1 - point0
    direction /= np.linalg.norm(direction)
    point0 += (distance / 2) * direction
    point1 -= (distance / 2) * direction
    return point0, point1


def top_down_orientation(gripper_open_direction) -> np.ndarray:
    X = gripper_open_direction / np.linalg.norm(gripper_open_direction)  # np.array([-1, 0, 0])
    Z = np.array([0, 0, -1])
    Y = np.cross(Z, X)
    return np.column_stack([X, Y, Z])

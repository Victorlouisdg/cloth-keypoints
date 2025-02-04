from typing import List

import cv2
import numpy as np
from camera_toolkit.reproject import project_world_to_image_plane
from cloth_manipulation.input_transform import InputTransform
from cloth_manipulation.motion_primitives.pull import ReorientTowelPull


class Panel:
    def __init__(self, image_buffer):
        self.image_buffer = image_buffer

    @staticmethod
    def fit_image_into_buffer(image, buffer, keep_aspect_ratio=True):
        panel_height, panel_width, _ = buffer.shape
        panel_aspect_ratio = float(panel_width) / float(panel_height)

        image_height, image_width, _ = image.shape
        image_aspect_ratio = float(image_width) / float(image_height)

        aspect_ratio_close = np.isclose(panel_aspect_ratio, image_aspect_ratio)

        if aspect_ratio_close or not keep_aspect_ratio:
            image = cv2.resize(image, (panel_width, panel_height))
            buffer[:, :, :] = image
            return buffer

        if image_aspect_ratio > panel_aspect_ratio:
            scale_factor = float(panel_width) / float(image_width)
            new_height = int(image_height * scale_factor)
            image = cv2.resize(image, (panel_width, new_height))
            padding_top = (panel_height - new_height) // 2
            buffer[padding_top : padding_top + new_height, :] = image
            return buffer
        else:
            scale_factor = float(panel_height) / float(image_height)
            new_width = int(image_width * scale_factor)
            image = cv2.resize(image, (new_width, panel_height))
            padding_left = (panel_width - new_width) // 2
            buffer[:, padding_left : padding_left + new_width, :] = image
            return buffer

    def fill_image_buffer(self, image, keep_aspect_ratio=True):
        self.image_buffer = Panel.fit_image_into_buffer(image, self.image_buffer, keep_aspect_ratio)


class FourPanels:
    def __init__(self, width: int = 1920, height: int = 1080):
        """HxWxC BGR"""
        rows = height
        columns = width
        middle_row = rows // 2
        middle_column = columns // 2
        self.image_buffer = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self.top_left = Panel(self.image_buffer[:middle_row, :middle_column])
        self.top_right = Panel(self.image_buffer[:middle_row, middle_column:])
        self.bottom_left = Panel(self.image_buffer[middle_row:, :middle_column])
        self.bottom_right = Panel(self.image_buffer[middle_row:, middle_column:])


def draw_center_circle(image) -> np.ndarray:
    h, w, _ = image.shape
    center_u = w // 2
    center_v = h // 2
    center = (center_u, center_v)
    image = cv2.circle(image, center, 1, (255, 0, 255), thickness=2)
    return image


def draw_cloth_transform_rectangle(image_full_size) -> np.ndarray:
    u_top = InputTransform.crop_start_u
    u_bottom = u_top + InputTransform.crop_width
    v_top = InputTransform.crop_start_v
    v_bottom = v_top + InputTransform.crop_height

    top_left = (u_top, v_top)
    bottom_right = (u_bottom, v_bottom)

    image = cv2.rectangle(image_full_size, top_left, bottom_right, (255, 0, 0), thickness=2)
    return image


def insert_transformed_into_original(original, transformed):
    u_top = InputTransform.crop_start_u
    u_bottom = u_top + InputTransform.crop_width
    v_top = InputTransform.crop_start_v
    v_bottom = v_top + InputTransform.crop_height

    transformed_unresized = cv2.resize(transformed, (InputTransform.crop_width, InputTransform.crop_height))
    original[
        v_top:v_bottom,
        u_top:u_bottom,
    ] = transformed_unresized


def draw_world_axes(image, world_to_camera, camera_matrix):
    origin = project_world_to_image_plane(np.zeros(3), world_to_camera, camera_matrix).astype(int)
    image = cv2.circle(image, origin.T, 10, (0, 255, 255), thickness=2)

    x_pos = project_world_to_image_plane([1.0, 0.0, 0.0], world_to_camera, camera_matrix).astype(int)
    x_neg = project_world_to_image_plane([-1.0, 0.0, 0.0], world_to_camera, camera_matrix).astype(int)
    y_pos = project_world_to_image_plane([0.0, 1.0, 0.0], world_to_camera, camera_matrix).astype(int)
    y_neg = project_world_to_image_plane([0.0, -1.0, 0.0], world_to_camera, camera_matrix).astype(int)
    image = cv2.line(image, x_pos.T, origin.T, color=(0, 0, 255), thickness=2)
    image = cv2.line(image, x_neg.T, origin.T, color=(100, 100, 255), thickness=2)
    image = cv2.line(image, y_pos.T, origin.T, color=(0, 255, 0), thickness=2)
    image = cv2.line(image, y_neg.T, origin.T, color=(150, 255, 150), thickness=2)

    z_pos = project_world_to_image_plane([0.0, 0.0, 1.0], world_to_camera, camera_matrix).astype(int)
    image = cv2.line(image, z_pos.T, origin.T, color=(255, 0, 0), thickness=2)
    return image


def draw_keypoints(image, keypoints: List[np.ndarray], world_to_camera, camera_matrix):
    for keypoint in keypoints:
        keypoint = project_world_to_image_plane(keypoint, world_to_camera, camera_matrix).astype(int)
        image = cv2.circle(image, keypoint, 1, color=(0, 0, 255), thickness=2)
    return image


def draw_pose(image, pose, world_to_camera, camera_matrix):
    pose_camera = world_to_camera @ pose
    rvec = pose_camera[:3, :3]
    tvec = pose_camera[:3, -1]
    image = cv2.drawFrameAxes(image, camera_matrix, np.zeros(4), rvec, tvec, 0.05)
    return image


def visualize_reorient_towel_pull(image, pull: ReorientTowelPull, world_to_camera, camera_matrix):
    def error_color_map(error):
        green = (0, 255, 0)
        red = (0, 0, 255)
        orange = (0, 150, 255)
        if error <= 0.05:
            return green
        elif error <= 0.1:
            return orange
        return red

    start = project_world_to_image_plane(pull.start, world_to_camera, camera_matrix).astype(int)
    end = project_world_to_image_plane(pull.end, world_to_camera, camera_matrix).astype(int)
    image = cv2.line(image, start.T, end.T, color=(255, 255, 0), thickness=2)

    draw_pose(image, pull.start_pose, world_to_camera, camera_matrix)
    draw_pose(image, pull.end_pose, world_to_camera, camera_matrix)

    def draw_corners(image, corners, color, draw_ids=False):
        corners_image = [
            project_world_to_image_plane(corner, world_to_camera, camera_matrix).astype(int) for corner in corners
        ]
        corners_image_for_polyline = np.array(corners_image, np.int32).reshape((-1, 1, 2))
        image = cv2.polylines(image, [corners_image_for_polyline], True, color, thickness=2)

        if draw_ids:
            for i, corner_image in enumerate(corners_image):
                font = cv2.FONT_HERSHEY_SIMPLEX
                image = cv2.putText(image, str(i), corner_image.T, font, 1, color, 2)

        return image

    image = draw_corners(image, pull.ordered_corners, (0, 255, 255))
    image = draw_corners(image, pull.centered_corners, (255, 255, 0))
    image = draw_corners(image, pull.desired_corners, (0, 255, 0))

    for corner, desired_corner in zip(pull.ordered_corners, pull.desired_corners):
        corner_image = project_world_to_image_plane(corner, world_to_camera, camera_matrix).astype(int)
        desired_corner_image = project_world_to_image_plane(desired_corner, world_to_camera, camera_matrix).astype(int)
        error = np.linalg.norm(desired_corner - corner)
        image = cv2.line(image, corner_image.T, desired_corner_image.T, color=error_color_map(error), thickness=1)

    average_error = pull.average_corner_error()
    text = f"Average corner error: {average_error:.3f} m"
    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(image, text, (50, 50), font, 1, error_color_map(average_error), 2)

    return image


def visualize_pick_reorient_towel_pull(
    image, grasp, pull: ReorientTowelPull, pull_end_pose, world_to_camera, camera_matrix
):
    def error_color_map(error):
        green = (0, 255, 0)
        red = (0, 0, 255)
        orange = (0, 150, 255)
        if error <= 0.05:
            return green
        elif error <= 0.1:
            return orange
        return red

    # start = project_world_to_image_plane(pull.start, world_to_camera, camera_matrix).astype(int)
    # end = project_world_to_image_plane(pull.end, world_to_camera, camera_matrix).astype(int)
    # image = cv2.line(image, start.T, end.T, color=(255, 255, 0), thickness=2)

    # draw_pose(image, grasp.get_pregrasp_pose(), world_to_camera, camera_matrix)

    def draw_corners(image, corners, color):
        corners_image = [project_world_to_image_plane(corner, world_to_camera, camera_matrix) for corner in corners]
        corners_image = np.array(corners_image, np.int32).reshape((-1, 1, 2))
        image = cv2.polylines(image, [corners_image], True, color, thickness=2)
        return image

    image = draw_corners(image, pull.ordered_corners, (0, 255, 255))
    image = draw_corners(image, pull.desired_corners, (0, 255, 0))

    for corner, desired_corner in zip(pull.ordered_corners, pull.desired_corners):
        project_world_to_image_plane(corner, world_to_camera, camera_matrix).astype(int)
        desired_corner_image = project_world_to_image_plane(desired_corner, world_to_camera, camera_matrix).astype(int)
        image = cv2.circle(image, desired_corner_image.T, 3, color=(0, 255, 0), thickness=4)

    start_image = project_world_to_image_plane(pull.start, world_to_camera, camera_matrix).astype(int)
    end_image = project_world_to_image_plane(pull.end, world_to_camera, camera_matrix).astype(int)
    cv2.line(image, start_image, end_image, color=(255, 0, 255), thickness=3)

    draw_pose(image, grasp.get_grasp_pose(), world_to_camera, camera_matrix)
    draw_pose(image, grasp.get_pregrasp_pose(), world_to_camera, camera_matrix)

    draw_pose(image, pull_end_pose, world_to_camera, camera_matrix)

    return image

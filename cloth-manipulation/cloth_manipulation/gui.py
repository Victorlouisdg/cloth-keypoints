import numpy as np
import cv2


class Panel:
    def __init__(self, image_buffer):
        self.image_buffer = image_buffer

    def fill_image_buffer(self, image, keep_aspect_ratio=True):
        panel_height, panel_width, _ = self.image_buffer.shape
        panel_aspect_ratio = float(panel_width) / float(panel_height)

        image_height, image_width, _ = image.shape
        image_aspect_ratio = float(image_width) / float(image_height)

        aspect_ratio_close = np.isclose(panel_aspect_ratio, image_aspect_ratio)

        if aspect_ratio_close or not keep_aspect_ratio:
            image = cv2.resize(image, (panel_width, panel_height))
            self.image_buffer[:, :, :] = image
            return

        if image_aspect_ratio > panel_aspect_ratio:
            scale_factor = float(panel_width) / float(image_width)
            new_height = int(image_height * scale_factor)
            image = cv2.resize(image, (panel_width, new_height))
            padding_top = (panel_height - new_height) // 2
            self.image_buffer[padding_top : padding_top + new_height, :] = image
        else:
            scale_factor = float(panel_height) / float(image_height)
            new_width = int(image_width * scale_factor)
            image = cv2.resize(image, (new_width, panel_height))
            padding_left = (panel_width - new_width) // 2
            self.image_buffer[:, padding_left : padding_left + new_width, :] = image


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

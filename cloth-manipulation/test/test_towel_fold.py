import pyzed.sl as sl
from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.camera_mapping import CameraMapping
from cloth_manipulation.controllers import FoldTowelController
from cloth_manipulation.hardware.setup_hardware import setup_victor_louise
from cloth_manipulation.manual_keypoints import get_manual_keypoints

victor_louise = setup_victor_louise()
victor_louise.dual_move_tcp(victor_louise.left.home_pose, victor_louise.right.home_pose)

resolution = sl.RESOLUTION.HD720
zed = Zed2i(resolution=resolution, serial_number=CameraMapping.serial_top, fps=30)

reorient_towel_controller = FoldTowelController(victor_louise)

image = zed.get_rgb_image()
corners = get_manual_keypoints(image)
reorient_towel_controller.act(corners)

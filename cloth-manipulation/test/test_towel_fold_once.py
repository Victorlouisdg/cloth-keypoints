from camera_toolkit.zed2i import Zed2i
from cloth_manipulation.hardware.setup_hw import setup_hw
from cloth_manipulation.towel import fold_towel_once

if __name__ == "__main__":
    dual_arm = setup_hw()
    # open ZED (and assert it is available)
    Zed2i.list_camera_serial_numbers()
    zed = Zed2i()
    fold_towel_once(zed, dual_arm)

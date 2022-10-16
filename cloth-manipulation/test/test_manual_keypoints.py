from pathlib import Path

import cv2
from cloth_manipulation.manual_keypoints import get_manual_keypoints

image_path = Path(__file__).parent.parent.parent / "resources" / "topdown_towel_720p.jpg"
image_original = cv2.imread(str(image_path))

keypoints = get_manual_keypoints(image_original)
print(f"{keypoints=}")

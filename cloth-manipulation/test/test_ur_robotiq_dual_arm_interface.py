import numpy as np
from scipy.spatial.transform import Rotation as R
from cloth_manipulation.setup_hw import setup_hw

dual_arm = setup_hw()


dual_arm.dual_moveL(pose_in_world_victor=dual_arm.victor_ur.home_pose, pose_in_world_louise=dual_arm.louise_ur.home_pose, vel = 2* dual_arm.victor_ur.DEFAULT_LINEAR_VEL)

victor = dual_arm.victor_ur

victor.moveL(victor.out_of_way_pose)

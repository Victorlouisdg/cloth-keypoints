import logging
import time

import numpy as np
from cloth_manipulation.motion_primitives.fold import execute_dual_fold_lines
from cloth_manipulation.motion_primitives.fold_trajectory_parameterization import CircularFoldTrajectory
from cloth_manipulation.setup_hw import setup_hw
################
# common world frame!
#################

logging.basicConfig(level=logging.DEBUG)

dual_arm = setup_hw()


fold_line_victor = CircularFoldTrajectory(np.array([0.-0.2, -0.2, 0.05]), np.array([-0.2, 0.2, 0.05]))
fold_line_louise = CircularFoldTrajectory(np.array([0.2, -0.2, 0.05]), np.array([0.2, 0.2, 0.05]))
print(fold_line_victor.get_fold_path(20))
# execute_single_fold_line(fold_line_victor,victor)
execute_dual_fold_lines(fold_line_victor, fold_line_louise, dual_arm)


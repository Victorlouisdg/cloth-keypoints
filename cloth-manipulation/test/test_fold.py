import logging

import numpy as np
from cloth_manipulation.motion_primitives.fold_execution import execute_dual_fold_trajectories, make_robot_formatted_trajectory_from_path, execute_single_fold_trajectory
from cloth_manipulation.motion_primitives.fold_trajectory_parameterization import CircularFoldTrajectory
from cloth_manipulation.hardware.setup_hw import setup_hw

################
# common world frame!
#################

logging.basicConfig(level=logging.DEBUG)

dual_arm = setup_hw()


fold_line_victor = CircularFoldTrajectory(np.array([0.0 - 0.1, -0.2, 0.0]), np.array([-0.1, 0.2, 0.0]))
fold_line_louise = CircularFoldTrajectory(np.array([0.1, -0.2, 0.0]), np.array([0.1, 0.2, 0.0]))
#print(make_robot_formatted_trajectory_from_path(fold_line_victor.get_fold_path(20)))
#execute_single_fold_line(fold_line_louise,dual_arm.louise_ur)
execute_dual_fold_trajectories(fold_line_victor, fold_line_louise, dual_arm)

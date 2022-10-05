from cloth_manipulation.setup_hw import setup_hw, home_waypoint_louise, home_victor_waypoint
from cloth_manipulation.motion_primitives.pull import execute_pull_primitive, select_towel_pull
import numpy as np
if __name__ == "__main__":
    dual_arm =setup_hw()
    dual_arm.dual_moveJ_IK(home_victor_waypoint,home_waypoint_louise)

    #pullprimitive = PullPrimitive(np.array([0,-0.1,0.1]),np.array([0,0.1,0.1]))

    # test towel on 4th quadrant

    pullprimitive = select_towel_pull([np.array([0.0,0.0,0.0]),np.array([0.23,0.0,0.0]),np.array([0.23,-0.3,0.0]),np.array([0.0,-0.3,0.0])])
    print(pullprimitive)
    execute_pull_primitive(pullprimitive,dual_arm.victor_ur)
    dual_arm.dual_moveJ_IK(home_victor_waypoint,home_waypoint_louise)
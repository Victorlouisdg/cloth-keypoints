import time

from cloth_manipulation.hardware.setup_hardware import setup_victor_louise

victor_louise = setup_victor_louise()

victor = victor_louise.left
louise = victor_louise.right


def check_gripper_and_motion(robot):
    # Check robot gripper
    robot.gripper.close()
    robot.gripper.open()

    # Check robot move to home
    robot.move_tcp(robot.home_pose)

    # Check robot move linear to out of way
    robot.move_tcp_linear(robot.out_of_way_pose, robot.LINEAR_SPEED, robot.LINEAR_ACCELERATION)


check_gripper_and_motion(victor)
check_gripper_and_motion(louise)

victor_louise.dual_gripper_close()
time.sleep(1)
victor_louise.dual_gripper_move_to_position(0.6)
time.sleep(1)
victor_louise.dual_gripper_open()


# Check synchronous movement
victor_louise.dual_move_tcp(victor.home_pose, louise.home_pose)
victor_louise.dual_move_tcp(victor.out_of_way_pose, louise.out_of_way_pose)

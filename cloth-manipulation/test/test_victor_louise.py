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
    robot.move_tcp_linear(robot.out_of_way_pose)


check_gripper_and_motion(victor)
check_gripper_and_motion(louise)

# Check synchronous movement
victor_louise.dual_move_tcp_linear(victor.home_pose, louise.home_pose)

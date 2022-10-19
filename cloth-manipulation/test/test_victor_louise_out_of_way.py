from cloth_manipulation.hardware.setup_hardware import setup_victor_louise

victor_louise = setup_victor_louise()

victor = victor_louise.left
louise = victor_louise.right

# Check synchronous movement
victor_louise.dual_gripper_move_to_position(0.7)
victor_louise.dual_move_tcp(victor.home_pose, louise.home_pose)
victor_louise.dual_move_tcp(victor.out_of_way_pose, louise.out_of_way_pose)

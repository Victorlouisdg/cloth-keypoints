from cloth_manipulation.motion_primitives.pull import TowelReorientPull

class ReorientTowelController():
    def __init__(self, dual_arms):
        self.dual_arms = dual_arms


    def act(self, keypoints):
        if len(keypoints) != 4:
            return

        pull = TowelReorientPull(keypoints, self.dual_arm)


        
        pass

    def visualize_plan(self, keypoints, image, world_to_camera, camera_matrix):
        pull = TowelReorientPull(keypoints, self.dual_arm)

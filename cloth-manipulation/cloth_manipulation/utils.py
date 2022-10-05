import numpy as np 

def angle_2D(v0, v1):
    #TODO: document. 
    x1, y1, *_ = v0
    x2, y2, *_ = v1
    dot = x1 * x2 + y1 * y2  # dot product between [x1, y1] and [x2, y2]
    det = x1 * y2 - y1 * x2  # determinant
    angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    return angle


def get_ordered_keypoints(keypoints):
    """
    orders keypoints according to their angle w.r.t. a frame that is created by translating the world frame to the center of the cloth.
    the first keypoints is the one with the smallest, positive angle and they are sorted counter-clockwise.
    """
    keypoints = np.array(keypoints)
    center = np.mean(keypoints, axis=1)

    angles = [angle_2D(center, keypoint) for keypoint in keypoints]
    angles = [angle % (2 * np.pi) for angle in angles]  # make angles positive from 0 to 2*pi
    keypoints_sorted = keypoints[np.argsort(angles)]
    return list(keypoints_sorted)
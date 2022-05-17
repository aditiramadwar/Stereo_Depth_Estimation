import numpy as np
import cv2
def getRotationTranslations(E):
    U, _, VT = np.linalg.svd(E)
    # S = ZW
    # U*Z*W*VT
    # Z = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 0]])
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    rot1 = (np.dot(U, np.dot(W, VT)))
    rot2 = rot1
    rot3 = (np.dot(U, np.dot(W.T, VT)))
    rot4 = rot3
    C1 = (U[:, 2])
    C2 = (-U[:, 2])
    C3 = C1
    C4 = C2
    R_right = [rot1, rot2, rot3, rot4]
    C_right = [C1, C2, C3, C4]
    for i in range(len(R_right)):
        if (np.linalg.det(R_right[i]) < 0):
            R_right[i] = -R_right[i]
            C_right[i] = -C_right[i]
    # for i in range(4):
    #     print(np.linalg.det(R_right[i]))
    pose = [[rot1, C1], [rot2, C2], [rot3, C3], [rot4, C4]]
    return pose

def DisambiguateCameraPose(left_pts, right_pts, k1, k2, right_pose):
    R_left = np.identity(3)
    C_left = np.zeros((3, 1)).reshape(3, 1)

    RT_left = np.dot(R_left, np.hstack((np.identity(3), -C_left)))
    P_left = np.dot(k1, RT_left)
    max_correct_z = 0
    best_i = 0
    al1_3d_pts = []
    for i, pose in enumerate(right_pose):
        RT_right = np.dot(pose[0], np.hstack((np.identity(3), -pose[1].reshape(3, 1))))
        P_right = np.dot(k2, RT_right)
        pts_4d = cv2.triangulatePoints(P_left, P_right, left_pts.T, right_pts.T)
        
        pts_3d = pts_4d[:-1]/pts_4d[-1, :]
        al1_3d_pts.append(pts_3d)
        correct_z = 0
        r3 = pose[0][2].reshape(1,-1)
        for idx in range(len(pts_3d[0])):
            cur_3d_pt = pts_3d[:, idx]
            if r3.dot(cur_3d_pt - pose[1]) > 0 and cur_3d_pt[2] > 0: 
                correct_z += 1

        if correct_z > max_correct_z:
            best_i = i
            max_correct_z = correct_z

    final_pose = right_pose[best_i]
    right_3d_pts = al1_3d_pts[best_i]
    return final_pose, right_3d_pts
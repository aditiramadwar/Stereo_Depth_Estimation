import cv2
from utils.EstimateFundamentalMatrix import *
def getMatches(img1, img2, n = -1, path = None):
    image1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    image2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=2000)
    train_keypoints, train_descriptor = sift.detectAndCompute(image1_gray, None)
    test_keypoints, test_descriptor = sift.detectAndCompute(image2_gray, None)

    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck = False)

    matches = bf.match(train_descriptor, test_descriptor)
    matches = sorted(matches, key = lambda x : x.distance)

    matches = matches[0:n]
    if path is not None:
        result = cv2.drawMatches(img1, train_keypoints, img2, test_keypoints, matches, img2, flags = 2)
        cv2.imwrite("results/"+path+"/all_matches_img.png", result)
        # cv2.imshow('matches', result)
        # cv2.waitKey(0) 
        # cv2.destroyAllWindows()

    xy = []
    for m in matches:
        pt1 = train_keypoints[m.queryIdx].pt
        pt2 = test_keypoints[m.trainIdx].pt
        xy.append([pt1[0], pt1[1], pt2[0], pt2[1]])
    xy = np.array(xy).reshape(-1, 4)
    return xy

def getInliers(all_matches):
    iterations = 1000
    # ideally it should come to zero (x1.T*F*x2 = 0)
    inlier_margin = 0.02
    max_inliers = 0
    final_inlier_idx = []
    final_F = 0
    for i in range(0,iterations):
        current_inliers = []
        n_row = all_matches.shape[0]
        idxs = np.random.choice(n_row, size=8)

        correspondences = all_matches[idxs, :] 
        F = getFundamentalMatrix(correspondences)
        # check of the matches are inliers for the F matrix, if yes then append them to the list
        for j in range(n_row):
            match_pair = all_matches[j]
            #(x1.T*F*x2 = 0)
            x1, x2 = match_pair[0:2], match_pair[2:4]
            # homogenous
            x1T = np.array([x1[0], x1[1], 1]).T
            x2_ = np.array([x2[0], x2[1], 1])
            margin = np.dot(x1T, np.dot(F, x2_))
            if np.abs(margin) < inlier_margin:
                current_inliers.append(j)
        
        # choose the matrix and matches that have the highest inliers
        if(max_inliers < len(current_inliers)):
            max_inliers = len(current_inliers)
            final_inlier_idx = current_inliers
            final_F = F

    final_matches = all_matches[final_inlier_idx, :]
    return final_F, final_matches

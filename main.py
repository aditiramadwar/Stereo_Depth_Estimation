from utils.GetInlierRANSANC import *
from utils.ExtractCameraPose import *
from utils.Rectification import *
import matplotlib.pyplot as plt
from utils.Depth import *
import argparse

Parser = argparse.ArgumentParser()
Parser.add_argument('--set', default='1', help='Dataset type 1-3, Default: 1')

Args = Parser.parse_args()
number = int(Args.set)
# Load the image
# number = 1
if(number==1):
    folder = 'curule'
    # thres = 10000
    k1 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    k2 = np.array([[1758.23, 0, 977.42], [0, 1758.23, 552.15], [0, 0, 1]])
    baseline = 88.39

elif(number==2):
    folder = 'octagon'
    k1 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    k2 = np.array([[1742.11, 0, 804.90], [0, 1742.11, 541.22], [0, 0, 1]])
    baseline = 221.76
    # thres = 100000
else:
    folder = 'pendulum'
    k1 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    k2 = np.array([[1729.05, 0, -364.24], [0, 1729.05, 552.22], [0, 0, 1]])
    baseline = 537.75
    # thres = 300000
print("Dataset selected:", folder)
f = k1[0,0]
image1 = cv2.imread('data/'+folder+'/im0.png')
image2 = cv2.imread('data/'+folder+'/im1.png')

image1 = cv2.resize(image1, (int(image1.shape[1] / 2), int(image1.shape[0] / 2)))
image2 = cv2.resize(image2, (int(image2.shape[1] / 2), int(image2.shape[0] / 2)))

################# Obtain feature matches ########################
matches_xy  = getMatches(image1, image2, 200, path = folder)
# print(len(matches_xy))
######################### RANSAC: Remove outliers, Fundamental Matrix with constraints #################
F, matches = getInliers(matches_xy)
print("Fundamental Matrix:\n", F)
print("Number of good matches found:", len(matches), "out of", len(matches_xy))

########################### Essenstial Matrix with constraints ########################

E = getEssensialMatrix(k1, k2, F)
print("Essential Matrix:\n", E)

########################### Get Rotation and Translation ##################
right_poses = getRotationTranslations(E)

left_pts = matches[:, 0:2]
right_pts = matches[:, 2:4]
final_pose, right_3d_pts = DisambiguateCameraPose(left_pts, right_pts, k1, k2, right_poses)
print("Final poses:\n R:\n", final_pose[0], "\nC:\n", final_pose[1])
####################### Rectification #####################
img1_warped, img2_warped = rectification(left_pts, right_pts, F,image1, image2, path = folder)
left_gray = cv2.cvtColor(img1_warped, cv2.COLOR_RGB2GRAY)
right_gray = cv2.cvtColor(img2_warped, cv2.COLOR_RGB2GRAY)

###################### Depth Map ###########################
disparity, depth = findDepthMap(left_gray, right_gray, f,baseline, window=5, Disprange = 50)

####################### Visualization ########################
fx, plts = plt.subplots(2,2,figsize = (15, 10))
plts[0][0].imshow(left_gray, cmap = 'gray')
plts[0][0].set_title('Gray Image')
plts[0][1].imshow(disparity, cmap=plt.cm.RdBu, interpolation='bilinear')
plts[0][1].set_title('RdBu disparity_map')
plts[1][0].imshow(disparity, cmap='gray', interpolation='bilinear')
plts[1][0].set_title('disp gray')
plts[1][1].imshow(disparity, cmap='hot', interpolation='nearest')
plts[1][1].set_title('disp heat')
plt.savefig("results/"+folder+"/DisparityMap.png")
plt.show()

fx, plts = plt.subplots(2,2,figsize = (15, 10))
plts[0][0].imshow(left_gray, cmap = 'gray')
plts[0][0].set_title('Gray Image')
plts[0][1].imshow(depth, cmap=plt.cm.RdBu, interpolation='bilinear')
plts[0][1].set_title('RdBu depth')
plts[1][0].imshow(depth, cmap='gray', interpolation='bilinear')
plts[1][0].set_title('depth gray')
plts[1][1].imshow(depth, cmap='hot', interpolation='nearest')
plts[1][1].set_title('depth heat')
plt.savefig("results/"+folder+"/DepthMap.png")
plt.show()

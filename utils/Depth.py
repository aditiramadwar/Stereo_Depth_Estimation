import numpy as np
import cv2
from tqdm import tqdm
def findDepthMap(left_gray, right_gray, f, baseline, window=5, Disprange = 50):
    disparity_map = getDisparity(left_gray, right_gray, f, baseline, window=window, Disprange = Disprange)
    
    disp_flat = disparity_map.flatten()
    disp_vals = np.unique(disp_flat)
    disp_vals_sort = np.sort(disp_vals)
    idx = int((15*len(disp_vals_sort))/100)
    disp_thres = disp_vals_sort[idx]
    print("disp thres:",disp_thres)

    thres = (baseline*f)/(disp_thres)
    print("depth thres:", thres)
    max_pixel = np.max(disparity_map)
    min_pixel = np.min(disparity_map)

    disparity_map_int = np.uint8(disparity_map * 255 / (max_pixel-min_pixel))

    depthMap = (baseline*f)/(disparity_map)
    print(np.max(depthMap))
    print(np.min(depthMap))
    depthMap[depthMap > thres] = thres
    depthMap = np.uint8(depthMap * 255 / (np.max(depthMap)-np.min(depthMap)))

    return disparity_map_int, depthMap

def getDisparity(left_gray, right_gray, f, baseline, window=5, Disprange = 50):
    left_gray = left_gray.astype(np.int32)
    right_gray = right_gray.astype(np.int32)
    height, width = left_gray.shape
    disparity_map = np.zeros((height, width))
    lim = Disprange
    # window = 10
    for y in tqdm(range(window, height - window)):
        for x in range(window, width - window):
            left_window = left_gray[y:y + window, x:x + window]
            min_dist = np.inf
            right_start = max(0, x - lim)
            right_end = min(width, x + lim)
            min_x = x
            for i in range(right_start,right_end):
                right_window = right_gray[y:y+window, i:i+window]    
                if left_window.shape != right_window.shape:
                    continue
                dist = np.sum(np.square(left_window - right_window))
                if(dist < min_dist):
                    min_dist = dist
                    min_x = i
            disparity_map[y, x] = np.abs(min_x - x)
    return disparity_map

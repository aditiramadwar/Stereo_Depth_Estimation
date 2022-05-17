import numpy as np
import cv2
def Epilines(lines,image, pts,  H = None):
    lines = lines.reshape(-1,3)
    size = image.shape[1]
    extended_pts = []
    for i, l in enumerate(lines):
        x_min = 0
        y_min = int(-l[2]/l[1])

        x_max = size
        y_max = int((-l[2]-l[0]*size)/l[1])

        epiline = [[x_min,y_min], [x_max,y_max]]

        extended_pts.append(epiline)  

    if H is not None:
        extended_pts = np.array(extended_pts)
        og_min = np.float32(extended_pts[:,0].reshape(-1,1,2))
        warp_min = cv2.perspectiveTransform(og_min, H).squeeze()

        og_max = np.float32(extended_pts[:,1].reshape(-1,1,2))
        warp_max = cv2.perspectiveTransform(og_max, H).squeeze()

        extended_pts = []
        for min, max in zip(warp_min, warp_max):
            extended_pts.append([min, max])

    extended_pts = np.array(extended_pts)
    img2_epi = draw(image, pts[:10], extended_pts)
    return img2_epi

def draw(image, points, linepts):
    draw_img = image.copy()
    for i, pt in enumerate(points):
        color = tuple(np.random.randint(0,255,3).tolist())
        x_min,y_min = linepts[i][0]
        x_max,y_max = linepts[i][1]
        draw_img = cv2.line(draw_img, (int(x_min),int(y_min)), (int(x_max),int(y_max)), color,1)

        x, y = pt[0], pt[1]
        draw_img = cv2.circle(draw_img, (int(x), int(y)), 5,  color, -1)
    return draw_img

def rectification(left_pts, right_pts, F, image1, image2, path = None):
    
    l1=cv2.computeCorrespondEpilines(right_pts.reshape(-1,1,2), 2, F)
    img2_epi = Epilines(l1, image1, left_pts)

    l2=cv2.computeCorrespondEpilines(left_pts.reshape(-1,1,2), 2, F)
    img1_epi = Epilines(l2, image2, right_pts)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    _, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(left_pts), np.float32(right_pts), F, imgSize=(w1, h1))
    im1_rectified = cv2.warpPerspective(image1, H1, (w1, h1))
    im2_rectified = cv2.warpPerspective(image2, H2, (w2, h2))

    dst1 = cv2.perspectiveTransform(left_pts.reshape(-1,1,2), H1).squeeze()
    dst2 = cv2.perspectiveTransform(right_pts.reshape(-1,1,2),H2).squeeze()

    warp_lines1 = cv2.computeCorrespondEpilines(right_pts.reshape(-1,1,2), 2, F)
    im1_print = Epilines(warp_lines1, im1_rectified, dst1, H = H1)

    warp_lines2 = cv2.computeCorrespondEpilines(left_pts.reshape(-1,1,2), 1, F)
    im2_print = Epilines(warp_lines2, im2_rectified, dst2, H = H2)

    if(path is not None):
        epilines_out = np.hstack((img1_epi, img2_epi))
        # cv2.imshow("epilines out", epilines_out)
        cv2.imwrite("results/"+path+"/epilines_img.png", epilines_out)
        # cv2.waitKey() 
        # cv2.destroyAllWindows()

        warped_out = np.hstack((im1_rectified, im2_rectified))
        # cv2.imshow("warped out", warped_out)
        cv2.imwrite("results/"+path+"/warped_img.png", warped_out)
        # cv2.waitKey() 
        # cv2.destroyAllWindows()

        out = np.hstack((im1_print, im2_print))
        cv2.imwrite("results/"+path+"/epi_warped_img.png", out)
        # cv2.imshow("warped epi", out)
        # cv2.waitKey() 
        # cv2.destroyAllWindows()

    return im1_rectified, im2_rectified 

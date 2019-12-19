import os
import cv2
import scipy.ndimage
import numpy as np

def remove_circles(img):
    circles = cv2.HoughCircles(img, method = cv2.HOUGH_GRADIENT, dp = 1, minDist = 1, param1 = 50, param2 = 5, minRadius = 0, maxRadius = 2)
    return circles

def draw_circles(circles, img):
    circles = circles[0]
    for circle in circles:
        x = circle[0]
        y = circle[1]
        r = circle[2]
        img = cv2.circle(img, (x, y), r, (255, 255, 255), 2)
    return img

#img_path = os.path.join("captchas", "0V18.jpg")
img_path = os.path.join("captchas", "2ORK.jpg")
print(img_path)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
ret, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
circles0 = remove_circles(img)
cv2.imshow("img", img)

erosion = ~img
erosion = cv2.erode(erosion, np.ones((2, 2), np.uint8), iterations = 1)
erosion = ~erosion
#cv2.imshow("erosion", erosion)
erosion = scipy.ndimage.median_filter(erosion, (5, 1))
#cv2.imshow("erosion - vert 5", erosion)
erosion = scipy.ndimage.median_filter(erosion, (1, 3))
#cv2.imshow("erosion - horz 3", erosion)
erosion2 = scipy.ndimage.median_filter(erosion, (1, 5))
#cv2.imshow("erosion - horz 5", erosion2)
erosion = cv2.erode(erosion, np.ones((2, 2), np.uint8), iterations = 1)
#cv2.imshow("dilation - horz 3", erosion)
erosion2 = cv2.erode(erosion2, np.ones((2, 2), np.uint8), iterations = 1)
#cv2.imshow("dilation - horz 5", erosion2)
erosion3 = scipy.ndimage.median_filter(erosion, (3, 3))
#cv2.imshow("dilation - filter 3", erosion3)
erosion4 = scipy.ndimage.median_filter(erosion2, (3, 3))
'''cv2.imshow("dilation2 - filter 3", erosion4)
erosion5 = scipy.ndimage.median_filter(erosion, (5, 5))
cv2.imshow("dilation - filter 5", erosion5)
erosion6 = scipy.ndimage.median_filter(erosion2, (5, 5))
cv2.imshow("dilation2 - filter 5", erosion6)'''
cv2.waitKey(0)

#img = scipy.ndimage.median_filter(img, (9, 1))
#img = scipy.ndimage.median_filter(img, (1, 5))
#cv2.imshow("filt img", img)

'''circles0 = remove_circles(img)
print(circles0)
if circles0 is not None:
    print(len(circles0[0]))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    newimg0 = draw_circles(circles0, img)
    cv2.imshow("newimg0", newimg0)'''
circles1 = remove_circles(erosion3)
print(circles1)
if circles1 is not None:
    print(len(circles1[0]))
    erosion3 = cv2.cvtColor(erosion3, cv2.COLOR_GRAY2RGB)
    cv2.imshow("erosion - circles - before", erosion3)
    erosion3 = draw_circles(circles1, erosion3)
    cv2.imshow("erosion - circles - after", erosion3)
circles2 = remove_circles(erosion4)
print(circles2)
if circles2 is not None:
    print(len(circles2[0]))
    erosion4 = cv2.cvtColor(erosion4, cv2.COLOR_GRAY2RGB)
    cv2.imshow("erosion2 - circles - before", erosion4)
    erosion4 = draw_circles(circles2, erosion4)
    cv2.imshow("erosion2 - circles - after", erosion4)

'''cv2.imshow("erosion3 - before", erosion3)
erosion3 = cv2.dilate(erosion3, np.ones((2, 2), np.uint8), iterations = 1) #scipy.ndimage.median_filter(erosion3, (3, 3, 3))
erosion3 = scipy.ndimage.median_filter(erosion3, (1, 3, 5))
erosion3 = scipy.ndimage.median_filter(erosion3, (3, 1, 5))
erosion3 = cv2.erode(erosion3, np.ones((2, 2), np.uint8), iterations = 1)
cv2.imshow("erosion3", erosion3)'''

cv2.imshow("erosion3 - before", erosion3)
erosion3 = cv2.dilate(erosion3, np.ones((3, 3), np.uint8), iterations = 1) #scipy.ndimage.median_filter(erosion3, (3, 3, 3))
erosion3 = scipy.ndimage.median_filter(erosion3, (5, 1, 5))
erosion3 = cv2.erode(erosion3, np.ones((3, 3), np.uint8), iterations = 2)
erosion3 = cv2.dilate(erosion3, np.ones((3, 3), np.uint8), iterations = 1)
cv2.imshow("erosion3", erosion3)

#medfilt_img = scipy.ndimage.median_filter(newimg0, (9, 1, 1))
#cv2.imshow("medilft img0", medfilt_img)
cv2.waitKey(0)

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
        img = cv2.circle(img, (x, y), r, (255, 255), 2)
    return img

img_path = os.path.join("captchas", "2ORK.jpg")
print(img_path)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
ret, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
circles0 = remove_circles(img)
cv2.imshow("img", img)

erosion = ~img
erosion = cv2.erode(erosion, np.ones((2, 2), np.uint8), iterations = 1)
erosion = ~erosion
erosion = scipy.ndimage.median_filter(erosion, (5, 1))
erosion = scipy.ndimage.median_filter(erosion, (1, 3))
erosion2 = scipy.ndimage.median_filter(erosion, (1, 5))
erosion = cv2.erode(erosion, np.ones((2, 2), np.uint8), iterations = 1)
erosion2 = cv2.erode(erosion2, np.ones((2, 2), np.uint8), iterations = 1)
erosion3 = scipy.ndimage.median_filter(erosion, (3, 3))

circles1 = remove_circles(erosion3)
if circles1 is not None:
    erosion3 = draw_circles(circles1, erosion3)

erosion3 = cv2.dilate(erosion3, np.ones((3, 3), np.uint8), iterations = 1) #scipy.ndimage.median_filter(erosion3, (3, 3, 3))
erosion3 = scipy.ndimage.median_filter(erosion3, (5, 1, 5))
erosion3 = cv2.erode(erosion3, np.ones((3, 3), np.uint8), iterations = 2)
erosion3 = cv2.dilate(erosion3, np.ones((3, 3), np.uint8), iterations = 1)
cv2.imshow("final product", erosion3)

cv2.waitKey(0)

print(erosion3.shape)
col_sums = erosion3.sum(axis = 0)
print(col_sums.shape)
print(col_sums)

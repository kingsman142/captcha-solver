import os
import cv2
import scipy.ndimage

def remove_circles(img):
    circles = cv2.HoughCircles(img, method = cv2.HOUGH_GRADIENT, dp = 2, minDist = 1, param1 = 100, param2 = 8, minRadius = 0, maxRadius = 3)
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
img_path = os.path.join("captchas", "0V18.jpg")
print(img_path)
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
ret, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
circles0 = remove_circles(img)
cv2.imshow("img", img)

img = scipy.ndimage.median_filter(img, (9, 1))
img = scipy.ndimage.median_filter(img, (1, 5))
cv2.imshow("filt img", img)

#laplacian = cv2.Laplacian(img, cv2.CV_64F)
#cv2.imshow("laplacian", laplacian)

gaussianblur1 = cv2.GaussianBlur(img, ksize = (3, 3), sigmaX = 5, sigmaY = 5)
gaussianblur2 = cv2.GaussianBlur(img, ksize = (5, 5), sigmaX = 5, sigmaY = 5)
gaussianblur3 = cv2.GaussianBlur(img, ksize = (7, 7), sigmaX = 5, sigmaY = 5)
gaussianblur4 = cv2.GaussianBlur(img, ksize = (3, 3), sigmaX = 10, sigmaY = 10)
gaussianblur5 = cv2.GaussianBlur(img, ksize = (5, 5), sigmaX = 10, sigmaY = 10)
gaussianblur6 = cv2.GaussianBlur(img, ksize = (7, 7), sigmaX = 10, sigmaY = 10)
'''cv2.imshow("gaussianblur1", gaussianblur1)
cv2.waitKey(0)
cv2.imshow("gaussianblur2", gaussianblur2)
cv2.waitKey(0)
#cv2.imshow("gaussianblur3", gaussianblur3)
#cv2.waitKey(0)
cv2.imshow("gaussianblur4", gaussianblur4)
cv2.waitKey(0)
cv2.imshow("gaussianblur5", gaussianblur5)
cv2.waitKey(0)
#cv2.imshow("gaussianblur6", gaussianblur6)
#cv2.waitKey(0)'''

circles0 = remove_circles(img)
print(circles0)
if circles0 is not None:
    print(len(circles0[0]))
'''circles1 = remove_circles(gaussianblur1)
print(circles1)
if circles1 is not None:
    print(len(circles1[0]))
circles2 = remove_circles(gaussianblur2)
print(circles2)
if circles2 is not None:
    print(len(circles2[0]))
circles3 = remove_circles(gaussianblur3)
print(circles3)
if circles3 is not None:
    print(len(circles3[0]))
circles4 = remove_circles(gaussianblur4)
print(circles4)
if circles4 is not None:
    print(len(circles4[0]))
circles5 = remove_circles(gaussianblur5)
print(circles5)
if circles5 is not None:
    print(len(circles5[0]))
circles6 = remove_circles(gaussianblur6)
print(circles6)
if circles6 is not None:
    print(len(circles6[0]))'''

if circles0 is not None:
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    newimg0 = draw_circles(circles0, img)
    cv2.imshow("newimg0", newimg0)
    #cv2.waitKey(0)
'''if circles1 is not None:
    gaussianblur1 = cv2.cvtColor(gaussianblur1, cv2.COLOR_GRAY2RGB)
    newimg1 = draw_circles(circles1, gaussianblur1)
    cv2.imshow("newimg1", newimg1)
    cv2.waitKey(0)
if circles2 is not None:
    gaussianblur2 = cv2.cvtColor(gaussianblur2, cv2.COLOR_GRAY2RGB)
    newimg2 = draw_circles(circles2, gaussianblur2)
    cv2.imshow("newimg2", newimg2)
    cv2.waitKey(0)
if circles3 is not None:
    gaussianblur3 = cv2.cvtColor(gaussianblur3, cv2.COLOR_GRAY2RGB)
    newimg3 = draw_circles(circles3, gaussianblur3)
    cv2.imshow("newimg3", newimg3)
    cv2.waitKey(0)
if circles4 is not None:
    gaussianblur4 = cv2.cvtColor(gaussianblur4, cv2.COLOR_GRAY2RGB)
    newimg4 = draw_circles(circles4, gaussianblur4)
    cv2.imshow("newimg4", newimg4)
    cv2.waitKey(0)
if circles5 is not None:
    gaussianblur5 = cv2.cvtColor(gaussianblur5, cv2.COLOR_GRAY2RGB)
    newimg5 = draw_circles(circles5, gaussianblur5)
    cv2.imshow("newimg5", newimg5)
    cv2.waitKey(0)
if circles6 is not None:
    gaussianblur6 = cv2.cvtColor(gaussianblur6, cv2.COLOR_GRAY2RGB)
    newimg6 = draw_circles(circles6, gaussianblur6)
    cv2.imshow("newimg6", newimg6)
    cv2.waitKey(0)'''

medfilt_img = scipy.ndimage.median_filter(newimg0, (9, 1, 1))
cv2.imshow("medilft img0", medfilt_img)
cv2.waitKey(0)

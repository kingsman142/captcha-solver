import os
import cv2
import scipy.ndimage
import numpy as np

class NoiseRemover():
    def _erase_circles(img, circles):
        circles = circles[0] # hough circles returns a nested list for some reason
        for circle in circles:
            x = circle[0] # x coordinate of circle's center
            y = circle[1] # y coordinate of circle's center
            r = circle[2] # radius of circle
            img = cv2.circle(img, center = (x, y), radius = r, color = (255), thickness = 2) # erase circle by making it white (to match the image background)
        return img

    def _detect_and_remove_circles(img):
        hough_circle_locations = cv2.HoughCircles(img, method = cv2.HOUGH_GRADIENT, dp = 1, minDist = 1, param1 = 50, param2 = 5, minRadius = 0, maxRadius = 2)
        if hough_circle_locations is not None:
            img = NoiseRemover._erase_circles(img, hough_circle_locations)
        return img

    def remove_all_noise(img):
        # run some basic tests to get rid of easy-to-remove noise -- first pass
        img = ~img # white letters, black background
        img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations = 1) # weaken circle noise and line noise
        img = ~img # black letters, white background
        img = scipy.ndimage.median_filter(img, (5, 1)) # remove line noise
        img = scipy.ndimage.median_filter(img, (1, 3)) # weaken circle noise
        img = cv2.erode(img, np.ones((2, 2), np.uint8), iterations = 1) # dilate image to initial stage (erode works similar to dilate because we thresholded the image the opposite way)
        img = scipy.ndimage.median_filter(img, (3, 3)) # remove any final 'weak' noise that might be present (line or circle)

        # detect any remaining circle noise
        img = NoiseRemover._detect_and_remove_circles(img) # after dilation, if concrete circles exist, use hough transform to remove them

        # eradicate any final noise that wasn't removed previously -- second pass
        img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations = 1) # actually performs erosion
        img = scipy.ndimage.median_filter(img, (5, 1)) # finally completely remove any extra noise that remains
        img = cv2.erode(img, np.ones((3, 3), np.uint8), iterations = 2) # dilate image to make it look like the original
        img = cv2.dilate(img, np.ones((3, 3), np.uint8), iterations = 1) # erode just a bit to polish fine details

        return img

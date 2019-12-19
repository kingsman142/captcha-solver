import glob
import os
import cv2

fns = glob.glob(os.path.join("captchas", "*.jpg"))
num_images = 20
for count, fn in enumerate(fns[0 : num_images]):
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("image {}/{}".format(count+1, num_images), img)
    cv2.waitKey(0)

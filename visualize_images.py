import glob
import os
import cv2
import sys

fns = glob.glob(os.path.join("data", "captchas", "*.jpg"))
num_images = sys.argv[1] if len(sys.argv) == 2 else 20
for count, fn in enumerate(fns[0 : num_images]):
    img = cv2.imread(fn, cv2.IMREAD_GRAYSCALE)
    cv2.imshow("image {}/{}".format(count+1, num_images), img)
    cv2.waitKey(0)

from captcha.image import ImageCaptcha
import cv2
import random
import string
import numpy as np
import sys # grab command-line arguments

def generateImage(length, width, height):
    label = ''.join(random.sample(string.uppercase + string.digits, length))
    image = ImageCaptcha(width, height)
    captcha = image.generate_image(label)
    return (captcha, label)

captchaLabels = []
numCaptchas = 100 if len(sys.argv) == 1 else int(sys.argv[1])
print "Generating ", numCaptchas, " captcha(s)"
for i in range(numCaptchas):
    captcha, label = generateImage(4, 300, 200)
    cv2.imwrite("./captchas/" + label + ".jpg", np.array(captcha))
    captchaLabels.append(label)

with open("./labels.txt", "w") as labelsFile:
    for label in captchaLabels:
        labelsFile.write(label + "\n")
    

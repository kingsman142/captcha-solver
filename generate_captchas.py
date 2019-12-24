from captcha.image import ImageCaptcha
import cv2
import random
import string
import os
import glob
import numpy as np
import sys # grab command-line arguments

def generateImage(length, width, height):
    letter_set = [chr(ascii_val) for ascii_val in range(ord('A'), ord('Z') + 1)]
    number_set = [chr(ascii_val) for ascii_val in range(ord('0'), ord('9') + 1)]
    char_set = letter_set + number_set

    label = ''.join(random.sample(char_set, length))
    image = ImageCaptcha(width, height)
    captcha = image.generate_image(label)
    return (captcha, label)

if not os.path.exists("data"):
    os.mkdir("data")
if not os.path.exists(os.path.join("data", "captchas")):
    os.mkdir(os.path.join("data", "captchas"))

captcha_paths = glob.glob(os.path.join("data", "captchas", "*.jpg"))
for path in captcha_paths:
    os.remove(path)

captcha_labels = []
num_captchas = 1000 if len(sys.argv) == 1 else int(sys.argv[1])
print("Generating {} captcha(s)".format(num_captchas))
for i in range(num_captchas):
    captcha, label = generateImage(4, 140, 76)
    captcha_path = os.path.join("data", "captchas", "{}.jpg".format(label))
    cv2.imwrite(captcha_path, np.array(captcha))
    captcha_labels.append(label)
    if (i+1) % 100 == 0:
        print("Generated {}/{} ({}%) CAPTCHAs...".format(i+1, num_captchas, round((i+1)/num_captchas * 100.0, 2)))
print("Finished generating captchas!")

with open("./labels.txt", "w") as labels_file:
    for label in captcha_labels:
        labels_file.write(label + "\n")

import os
import cv2
import glob
import shutil
import scipy.ndimage
import numpy as np

from noise_remover import NoiseRemover
from character_segmenter import CharacterSegmenter

# keep track of which characters we're trying to classify
letter_set = [chr(ascii_val) for ascii_val in range(ord('A'), ord('Z') + 1)]
number_set = [chr(ascii_val) for ascii_val in range(ord('0'), ord('9') + 1)]
char_set = letter_set + number_set
char_counts = {char : 0 for char in char_set}

if os.path.exists(os.path.join("data", "characters", "all_chars")):
    shutil.rmtree(os.path.join("data", "characters", "all_chars")) # clear previous data

# check if 'characters' folder exists, as well as folders for each digit individually
characters_dataset_path = os.path.join("data", "characters")
if not os.path.exists(characters_dataset_path):
    os.mkdir(characters_dataset_path)
characters_dataset_split_path = os.path.join("data", "characters", "all_chars")
if not os.path.exists(characters_dataset_split_path):
    os.mkdir(characters_dataset_split_path)
for char in char_set:
    char_folder_path = os.path.join("data", "characters", "all_chars", char)
    if not os.path.exists(char_folder_path):
        os.mkdir(char_folder_path)

# loop over all the CAPTCHA images
captchas_path = os.path.join("data", "captchas", "*.jpg") # path to all CAPTCHAs
captcha_paths = glob.glob(captchas_path) # path to individual CAPTCHAs
num_bad_captchas = 0
for captcha_index, captcha_path in enumerate(captcha_paths):
    # image meta-details
    img_fn = os.path.split(captcha_path)[1] # convert from "data/captchas/1ZX0.jpg" to "1ZX0.jpg"
    captcha_label = img_fn.split(".")[0] # convert from "1ZX0.jpg" to "1ZX0"

    # read in image and perform preliminary thresholding to prepare for denoising
    img = cv2.imread(captcha_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)

    # clean up the image by removing noise
    clean_image = NoiseRemover.remove_all_noise(img)

    masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs = CharacterSegmenter.get_components(clean_image)
    if len(masks) == 0:
        num_bad_captchas += 1
        continue

    # segment and extract characters
    masks, mask_start_indices = CharacterSegmenter.segment_characters(masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs)
    if not len(masks) == 4:
        num_bad_captchas += 1
        continue

    # reorder masks and starting indices in ascending order to align them with the proper character for labeling
    mask_start_indices, indices = zip(*sorted(zip(mask_start_indices, [i for i in range(len(mask_start_indices))]))) # make sure intervals are in left-to-right order so we can segment characters properly
    masks = [masks[i] for i in indices]
    char_infos = [(masks[i], captcha_label[i]) for i in range(len(masks))]

    # save characters to disk
    for index, char_info in enumerate(char_infos):
        char_crop, label = char_info

        # reshape character crop to 76x76
        char_crop = CharacterSegmenter.squarify_image(char_crop)
        char_crop = ~char_crop

        # save digit to file so we can train a CNN later
        char_save_path = os.path.join("data", "characters", "all_chars", label, "{}_{}.jpg".format(label, char_counts[label]))
        cv2.imwrite(char_save_path, char_crop)
        char_counts[label] += 1

    if captcha_index % 100 == 0:
        print("Processed {}/{} ({}%) CAPTCHAs...".format(captcha_index + 1, len(captcha_paths), round((captcha_index+1) / len(captcha_paths) * 100.0, 2)))
print("Number of bad CAPTCHAs: {}/{} ({}%)".format(num_bad_captchas, len(captcha_paths), num_bad_captchas / len(captcha_paths) * 100.0))

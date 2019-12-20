import os
import cv2
import math
import glob
import shutil
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

class CharacterSegmenter():
    def find_nonzero_intervals(vec):
        zero_elements = (vec == 0) * 1 # mark zero-elements as 1 and non-zero elements as 0
        nonzero_borders = np.diff(zero_elements) # find diff between each element and its neighbor (-1 and 1 represent borders, 0 represents segment or non-segment element)
        edges, = np.nonzero(nonzero_borders) # NOTE: comma is vital to extract first element from tuple
        edge_vec = [edges+1] # helps maintain zero-indexing properties (not important to discuss)
        if vec[0] != 0: # special case: catch a segment that starts at the beginning of the array without a 0 border
            edge_vec.insert(0, [0]) # index 0 goes at the beginning of the list to remain proper spatial ordering of intervals
        if vec[-1] != 0: # special case: catch a segment that ends at the end of the array without a 0 border
            edge_vec.append([len(vec)]) # goes at the end of the list to remain proper spatial ordering of intervals
        edges = np.concatenate(edge_vec) # generate final edge list containing indices of 0 elements bordering non-zero segments
        interval_pairs = [(edges[i], edges[i+1]) for i in range(0, len(edges)-1, 2)] # pair up start and end indices
        interval_lengths = [pair[1] - pair[0] for pair in interval_pairs]
        return interval_pairs, interval_lengths

    def get_components(img):
        # find number of components
        img = ~img
        ret, markers_original = cv2.connectedComponents(img)

        placeholder = ~img

        # perform watershed segmentation
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        markers = cv2.watershed(~img, markers_original)

        markers[placeholder == 255] = -1
        unique_markers = np.unique(markers)

        masks = []
        mask_sizes = []
        mask_start_indices = []
        mask_char_pixels_arrs = []
        if len(unique_markers) > 1:
            for marker in unique_markers[1:]:
                # extract the mask
                mask = np.array((markers != marker) * 255, np.uint8)
                image_height = mask.shape[0]
                num_white_pixels = (mask.sum(axis = 0) / 255) # count number of 255-valued (white) pixels in each column
                num_char_pixels = image_height - num_white_pixels
                mask_start_index = np.nonzero(num_char_pixels > 0)[0][0]

                # crop image to only tightly wrap around the character
                interval, _ = CharacterSegmenter.find_nonzero_intervals(num_char_pixels)
                start, end = interval[0]
                mask = mask[:, start:end]

                # count number of pixels corresponding to a character in the image
                char_pixels_arr = np.count_nonzero(mask == 0, axis = 0)
                num_black_pixels = np.sum(char_pixels_arr)

                # meta information about the mask
                masks.append(mask)
                mask_sizes.append(num_black_pixels)
                mask_start_indices.append(mask_start_index)
                mask_char_pixels_arrs.append(char_pixels_arr)
        return (masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs)

    def segment_characters(masks, mask_sizes, mask_start_indices, mask_char_pixel_arrs):
        # prune out characters with too few pixels (they're just noise)
        masks = [masks[i] for i in range(len(masks)) if mask_sizes[i] > 100]
        mask_start_indices = [mask_start_indices[i] for i in range(len(mask_start_indices)) if mask_sizes[i] > 100]
        mask_char_pixel_arrs = [mask_char_pixel_arrs[i] for i in range(len(mask_char_pixel_arrs)) if mask_sizes[i] > 100]
        mask_sizes = [size for size in mask_sizes if size > 100]

        while len(masks) < 4: # while we haven't found 4 intervals representing 4 characters
            largest_mask_index = np.argmax(mask_sizes) # index of longest interval (split up largest interval because it's the most likely one to have more than one character)

            largest_mask = masks[largest_mask_index]
            largest_mask_size = mask_sizes[largest_mask_index]
            mask_start_index = mask_start_indices[largest_mask_index] # unwrap interval tuple
            mask_char_pixels = mask_char_pixel_arrs[largest_mask_index]

            # when splitting up an interval that might contain 2 characters, we COULD split it directly down the middle, but that's a naive approach
            # instead, just say the best candidate column index to split the characters is the column with the fewest black pixels that's in the middle of this interval (if you include the edges, those might be labeled as the 'best candidate', when in reality they're just the beginning or end edge of a character, and not at the intersection of the two characters)
            padding_value = 0.49 if largest_mask_size < 2200 else 0.1
            margin_length = int(largest_mask.shape[1] * padding_value) # only consider candidates in the middle (padding_value)% of the interval (to remove noisy results on edges of characters), so remove 25% of the interval to the left and 25% of the interval to the right
            new_interval_start = margin_length # start index in the middle (padding_value)% of this interval
            new_interval_end = largest_mask.shape[1] - margin_length # end index in the middle (padding_value)% of this interval
            divider_offset = np.argmin(mask_char_pixels[new_interval_start : new_interval_end]) # found the best candidate column to split the characters -- call this the offset of the character divider from the true start index of the interval

            # preprocess left sub-mask
            left_start = 0
            left_end = new_interval_start + divider_offset
            left_mask = largest_mask[:, left_start : left_end]
            left_char_pixels = mask_char_pixels[left_start : left_end]
            left_start_index = mask_start_index
            left_mask_size = np.sum(left_char_pixels)

            # preprocess right sub-mask
            right_start = new_interval_start + divider_offset
            right_end = largest_mask.shape[1]
            right_mask = largest_mask[:, right_start : right_end]
            right_char_pixels = mask_char_pixels[right_start : right_end]
            right_start_index = mask_start_index + new_interval_start + divider_offset
            right_mask_size = np.sum(right_char_pixels)

            # replace the 'super-interval' (most likely containing two characters) in the intervals list with the two new sub-intervals
            masks[largest_mask_index] = left_mask
            masks.insert(largest_mask_index + 1, right_mask)
            mask_sizes[largest_mask_index] = left_mask_size
            mask_sizes.insert(largest_mask_index + 1, right_mask_size)
            mask_start_indices[largest_mask_index] = left_start_index
            mask_start_indices.insert(largest_mask_index + 1, right_start_index)
            mask_char_pixel_arrs[largest_mask_index] = left_char_pixels
            mask_char_pixel_arrs.insert(largest_mask_index + 1, right_char_pixels)

            # prune out characters with too few pixels (they're just noise)
            masks = [masks[i] for i in range(len(masks)) if mask_sizes[i] > 100]
            mask_start_indices = [mask_start_indices[i] for i in range(len(mask_start_indices)) if mask_sizes[i] > 100]
            mask_char_pixel_arrs = [mask_char_pixel_arrs[i] for i in range(len(mask_char_pixel_arrs)) if mask_sizes[i] > 100]
            mask_sizes = [size for size in mask_sizes if size > 100]
        return masks, mask_start_indices

# keep track of which characters we're trying to classify
letter_set = [chr(ascii_val) for ascii_val in range(ord('A'), ord('Z') + 1)]
number_set = [chr(ascii_val) for ascii_val in range(ord('0'), ord('9') + 1)]
char_set = letter_set + number_set
char_counts = {char : 0 for char in char_set}

shutil.rmtree(os.path.join("data", "characters")) # clear previous data

# check if 'characters' folder exists, as well as folders for each digit individually
characters_dataset_path = os.path.join("data", "characters")
if not os.path.exists(characters_dataset_path):
    os.mkdir(characters_dataset_path)
for char in char_set:
    char_folder_path = os.path.join("data", "characters", char)
    if not os.path.exists(char_folder_path):
        os.mkdir(char_folder_path)

# loop over all the CAPTCHA images
captchas_path = os.path.join("data", "captchas", "*.jpg") # path to all CAPTCHAs
captcha_paths = glob.glob(captchas_path) # path to individual CAPTCHAs
for captcha_path in captcha_paths:
    # image meta-details
    img_fn = os.path.split(captcha_path)[1] # convert from "data/captchas/1ZX0.jpg" to "1ZX0.jpg"
    captcha_label = img_fn.split(".")[0] # convert from "1ZX0.jpg" to "1ZX0"

    # read in image and perform preliminary thresholding to prepare for denoising
    img = cv2.imread(captcha_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)

    # clean up the image by removing noise
    clean_image = NoiseRemover.remove_all_noise(img)

    masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs = CharacterSegmenter.get_components(clean_image)

    # segment and extract characters
    masks, mask_start_indices = CharacterSegmenter.segment_characters(masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs)

    # reorder masks and starting indices in ascending order to align them with the proper character for labeling
    mask_start_indices, masks = zip(*sorted(zip(mask_start_indices, masks))) # make sure intervals are in left-to-right order so we can segment characters properly
    char_infos = [(masks[i], captcha_label[i]) for i in range(len(masks))]

    # save characters to disk
    for index, char_info in enumerate(char_infos):
        char_crop, label = char_info

        # reshape character crop to 76x76
        crop_width = char_crop.shape[1]
        padding = (76 - crop_width) / 2 # force crop to go from 76 x crop_width to 76 x 76 so we can train a CNN
        char_crop = cv2.copyMakeBorder(char_crop, top = 0, bottom = 0, left = math.floor(padding), right = math.ceil(padding), borderType = cv2.BORDER_CONSTANT, value = 255)

        # save digit to file so we can train a CNN later
        char_save_path = os.path.join("data", "characters", label, "{}.jpg".format(char_counts[label]))
        cv2.imwrite(char_save_path, char_crop)
        char_counts[label] += 1

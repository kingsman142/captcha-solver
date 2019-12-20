import os
import cv2
import math
import glob
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

    def _num_black_pixels_in_crops(img, intervals):
        interval_sizes = []
        for index, interval in enumerate(intervals):
            start, end = interval
            char_crop = img[:, start:end]
            num_black_pixels = np.count_nonzero(char_crop == 0)
            interval_sizes.append(num_black_pixels)
        return interval_sizes

    def visualize_char_dividers(img, intervals):
        img_copy = img.copy()
        for interval in intervals:
            start, end = interval
            cv2.line(img_copy, (start, 0), (start, img_copy.shape[1]), (0), thickness=1, lineType=8)
            cv2.line(img_copy, (end, 0), (end, img_copy.shape[1]), (0), thickness=1, lineType=8)
        cv2.imshow("Character Dividers", img_copy)

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

                # transform the crop to 76x76
                #mask_width = mask.shape[1]
                #padding = (76 - mask_width) / 2 # force crop to go from 76 x crop_width to 76 x 76 so we can train a CNN
                #mask = cv2.copyMakeBorder(mask, top = 0, bottom = 0, left = math.floor(padding), right = math.ceil(padding), borderType = cv2.BORDER_CONSTANT, value = 255)
                char_pixels_arr = np.count_nonzero(mask == 0, axis = 0)
                num_black_pixels = np.sum(char_pixels_arr)
                print("printing num black pixels")
                print(num_black_pixels)

                # meta information about the mask
                masks.append(mask)
                mask_sizes.append(num_black_pixels)
                mask_start_indices.append(mask_start_index)
                mask_char_pixels_arrs.append(char_pixels_arr)
                cv2.imshow("marker {}, size {}, start index {}".format(marker, num_black_pixels, mask_start_index), mask)
        return (masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs)

    def find_char_boundaries(masks, mask_sizes, mask_start_indices, mask_char_pixel_arrs):
        # prune out characters with too few pixels (they're just noise)
        print("1", mask_sizes, mask_start_indices)
        masks = [masks[i] for i in range(len(masks)) if mask_sizes[i] > 100]
        mask_start_indices = [mask_start_indices[i] for i in range(len(mask_start_indices)) if mask_sizes[i] > 100]
        mask_char_pixel_arrs = [mask_char_pixel_arrs[i] for i in range(len(mask_char_pixel_arrs)) if mask_sizes[i] > 100]
        mask_sizes = [size for size in mask_sizes if size > 100]
        print("2", mask_sizes, mask_start_indices)

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
            print("left start, left end", left_start, left_end)
            left_mask = largest_mask[:, left_start : left_end]
            #left_mask_width = left_mask.shape[1]
            #left_mask_padding = (76 - left_mask_width) / 2 # force crop to go from 76 x crop_width to 76 x 76 so we can train a CNN
            #left_mask = cv2.copyMakeBorder(left_mask, top = 0, bottom = 0, left = math.floor(left_mask_padding), right = math.ceil(left_mask_padding), borderType = cv2.BORDER_CONSTANT, value = 255)
            left_char_pixels = mask_char_pixels[left_start : left_end]
            left_start_index = mask_start_index
            left_mask_size = np.sum(left_char_pixels)
            print(left_char_pixels)
            print(left_mask_size)
            cv2.imshow("left mask", left_mask)

            # preprocess right sub-mask
            right_start = new_interval_start + divider_offset
            right_end = largest_mask.shape[1]
            print("right start, right end", right_start, right_end)
            right_mask = largest_mask[:, right_start : right_end]
            #right_mask_width = right_mask.shape[1]
            #right_mask_padding = (76 - right_mask_width) / 2 # force crop to go from 76 x crop_width to 76 x 76 so we can train a CNN
            #right_mask = cv2.copyMakeBorder(right_mask, top = 0, bottom = 0, left = math.floor(right_mask_padding), right = math.ceil(right_mask_padding), borderType = cv2.BORDER_CONSTANT, value = 255)
            right_char_pixels = mask_char_pixels[right_start : right_end]
            right_start_index = mask_start_index + new_interval_start + divider_offset
            right_mask_size = np.sum(right_char_pixels)
            print(right_char_pixels)
            print(right_mask_size)
            cv2.imshow("right mask", right_mask)

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
            print("3", mask_sizes, mask_start_indices)
            masks = [masks[i] for i in range(len(masks)) if mask_sizes[i] > 100]
            mask_start_indices = [mask_start_indices[i] for i in range(len(mask_start_indices)) if mask_sizes[i] > 100]
            mask_char_pixel_arrs = [mask_char_pixel_arrs[i] for i in range(len(mask_char_pixel_arrs)) if mask_sizes[i] > 100]
            mask_sizes = [size for size in mask_sizes if size > 100]
            print("4", mask_sizes, mask_start_indices)
            cv2.waitKey(0)
        return masks, mask_start_indices

    def segment_characters(img, captcha_label, intervals):
        output = []
        for index, interval in enumerate(intervals):
            start_column, end_column = interval
            char_crop = clean_image[:, start_column : end_column] # crop out the character from the image
            crop_width = char_crop.shape[1] # character width

            # padding represents how many columns we need to add to this cropped image to make it 76x76
            # divide this above value by 2 to find out how much padding to add to the left side of the image, and how much to add to the right side
            # on the below line with cv2.copyMakeBorder(...), we use floor() and ceil() to make sure 'left' and 'right' add up to the value of padding, since padding might be a decimal
            padding = (76 - crop_width) / 2 # force crop to go from 76 x crop_width to 76 x 76 so we can train a CNN
            char_crop = cv2.copyMakeBorder(char_crop, top = 0, bottom = 0, left = math.floor(padding), right = math.ceil(padding), borderType = cv2.BORDER_CONSTANT, value = 255) # extend left and right border of cropped image by adding columns of white pixels

            label = captcha_label[index]
            cv2.imshow("char {}".format(index), char_crop)
            output.append((char_crop, label))
        return output

# keep track of which digits we're trying to classify
letter_set = [chr(ascii_val) for ascii_val in range(ord('A'), ord('Z') + 1)]
number_set = [chr(ascii_val) for ascii_val in range(ord('0'), ord('9') + 1)]
char_set = letter_set + number_set

# check if 'digits' folder exists, as well as folders for each digit individually
digits_dataset_path = os.path.join("data", "digits")
if not os.path.exists(digits_dataset_path):
    os.mkdir(digits_dataset_path)
for char in char_set:
    char_folder_path = os.path.join("data", "digits", char)
    if not os.path.exists(char_folder_path):
        os.mkdir(char_folder_path)

# loop over all the CAPTCHA images
captchas_path = os.path.join("data", "captchas", "*.jpg") # path to all CAPTCHAs
captcha_paths = glob.glob(captchas_path) # path to individual CAPTCHAs
for captcha_path in captcha_paths:
    # image meta-details
    img_fn = os.path.split(captcha_path)[1] # convert from "data/captchas/1ZX0.jpg" to "1ZX0.jpg"
    captcha_label = img_fn.split(".")[0] # convert from "1ZX0.jpg" to "1ZX0"
    print(img_fn, captcha_label)

    # read in image and perform preliminary thresholding to prepare for denoising
    img = cv2.imread(captcha_path, cv2.IMREAD_GRAYSCALE)
    _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)

    # clean up the image by removing noise
    clean_image = NoiseRemover.remove_all_noise(img)
    cv2.imshow("clean image", clean_image)

    print(np.count_nonzero(clean_image == 0, axis = 0))
    masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs = CharacterSegmenter.get_components(clean_image)
    for index, mask in enumerate(masks):
        cv2.imshow("mask {}".format(index), mask)
    print(mask_sizes, mask_start_indices)
    print(mask_char_pixels_arrs)

    # segment and extract characters
    masks, mask_start_indices = CharacterSegmenter.find_char_boundaries(masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs)
    print(mask_start_indices)
    mask_start_indices, masks = zip(*sorted(zip(mask_start_indices, masks))) # make sure intervals are in left-to-right order so we can segment characters properly
    print(mask_start_indices)
    char_infos = [(masks[i], captcha_label[i]) for i in range(len(masks))]
    for index, char_info in enumerate(char_infos):
        cv2.imshow("char {}: {}".format(index, char_info[1]), char_info[0])

    # OPTIONAL (uncomment to use): used for visualization purposes, to see a bunch of vertical lines separating characters in the image
    # CharacterSegmenter.visualize_dividers(clean_image, intervals)

    cv2.waitKey(0)

'''captcha_label = "0V18"
img_path = os.path.join("captchas", "0V18.jpg")
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
_, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY)
cv2.imshow("img", img)'''

# optional -- i think it produces better output than above "final product"
'''clean_image = ~(~img - (clean_image))
clean_image = cv2.GaussianBlur(clean_image, (3, 3), sigmaX = 1)
kernel = np.array([[-1,-1,-1], [-1,10,-1], [-1,-1,-1]])
clean_image = cv2.filter2D(clean_image, -1, kernel)
clean_image = cv2.erode(clean_image, np.ones((2, 2), np.uint8), iterations = 1)
cv2.imshow("masked", clean_image)'''

#cv2.imwrite("test.jpg", clean_image)
cv2.waitKey(0)

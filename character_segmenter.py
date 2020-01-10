import cv2
import numpy as np
import math

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

    def squarify_image(img):
        # reshape character crop from (height x width) to (height x height) where height > width
        img_height, img_width = img.shape
        if img_height > img_width: # make the image fatter
            padding = (img_height - img_width) / 2 # force crop to go from 76 x crop_width to 76 x 76 so we can train a CNN
            img = cv2.copyMakeBorder(img, top = 0, bottom = 0, left = math.floor(padding), right = math.ceil(padding), borderType = cv2.BORDER_CONSTANT, value = 255)
        elif img_height < img_width: # make the image skinnier
            margin = (img_width - img_height) / 2
            begin_column = int(0 + math.floor(margin))
            end_column = int(img_width - math.ceil(margin))
            img = img[:, begin_column : end_column]
        return img

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

        iters = 0
        while len(masks) < 4 and len(masks) > 0 and iters < 10: # while we haven't found 4 intervals representing 4 characters
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

            iters += 1
        return masks, mask_start_indices

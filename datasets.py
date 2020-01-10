import os
import cv2
import torch
import random
import glob
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms

from noise_remover import NoiseRemover
from character_segmenter import CharacterSegmenter

class CharactersDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, validate = False):
        super(CharactersDataset, self).__init__()

        self.data_root = data_root
        self.validate = validate

        image_search_path = os.path.join(data_root, "**", "*.jpg")
        self.image_list = glob.glob(image_search_path, recursive = True)

        letter_set = [chr(ascii_val) for ascii_val in range(ord('A'), ord('Z') + 1)]
        number_set = [chr(ascii_val) for ascii_val in range(ord('0'), ord('9') + 1)]
        self.char_set = letter_set + number_set

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomAffine(degrees = 45, translate = (0.2, 0.2), scale = (0.7, 1.3), fillcolor = 0)
        ])

    def _load_image(self, index):
        img_path = self.image_list[index]
        img_fn = os.path.split(img_path)[1]
        img_label = img_fn.split(".")[0].split("_")[0] # convert from "A_1385.jpg" to "A_1385" to "A"

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = img[None, :, :]
        img = img = cv2.normalize(img, None, alpha = 0, beta = 1, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F) # change pixel range from [0, 255] to [0, 1] # change pixel range from [0, 255] to [0, 1]
        img *= 2.0
        img -= 1.0

        return img, img_label

    def __getitem__(self, index):
        img, label = self._load_image(index)
        while img is None:
            index = random.randint(0, self.__len__())
            img, label = self._load_image(index)
        label = self.char_set.index(label)

        img = torch.Tensor(img)#self.transform(img) if self.validate else torch.Tensor(img)
        label = torch.as_tensor(label)
        return {'imgs': img, 'labels': label}

    def __len__(self):
        return len(self.image_list)

class CAPTCHADataset(torch.utils.data.Dataset):
    def __init__(self, data_root, img_format = "{}.jpg", size = -1):
        super(CAPTCHADataset, self).__init__()

        # root directory where the CAPTCHAs live
        self.data_root = data_root

        image_dir = os.path.join(self.data_root, "*.jpg")
        self.image_list = glob.glob(image_dir)
        if size > 0: # if we're only selecting a subset of the test set, select a random subset
            random.shuffle(self.image_list)
            self.image_list = self.image_list[0:size]

        # let us convert from ASCII labels to integer labels later on
        letter_set = [chr(ascii_val) for ascii_val in range(ord('A'), ord('Z') + 1)]
        number_set = [chr(ascii_val) for ascii_val in range(ord('0'), ord('9') + 1)]
        self.char_set = letter_set + number_set

    def _load_image(self, index):
        # find image path and label indicating which characters are in the CAPTCHA
        img_path = self.image_list[index]
        captcha_label = os.path.split(img_path)[1].split(".")[0] # convert from 'data/38A7.jpg' to '38A7.jpg' to '38A7'

        # load the image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        ### preprocess the image (same steps used in preprocess_dataset.py)
        _, img = cv2.threshold(img, 230, 255, cv2.THRESH_BINARY) # binarize the image
        clean_image = NoiseRemover.remove_all_noise(img) # clean up the image by removing noise

        # find how many characters there might be to see if we need to extract additional data
        masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs = CharacterSegmenter.get_components(clean_image)
        if len(masks) == 0:
            return None, None

        # segment and extract characters
        masks, mask_start_indices = CharacterSegmenter.segment_characters(masks, mask_sizes, mask_start_indices, mask_char_pixels_arrs)
        if not len(masks) == 4:
            return None, None

        # reorder masks and starting indices in ascending order to align them with the proper character for labeling
        mask_start_indices, indices = zip(*sorted(zip(mask_start_indices, [i for i in range(len(mask_start_indices))]))) # make sure intervals are in left-to-right order so we can segment characters properly
        masks = [masks[i] for i in indices]

        # split chars and labels into two separate lists
        chars = [masks[i] for i in range(len(masks))]
        labels = [captcha_label[i] for i in range(len(masks))]

        # reshape character crops to 76x76
        chars = [CharacterSegmenter.squarify_image(char) for char in chars]
        chars = [~char for char in chars]

        return chars, labels

    def __getitem__(self, index):
        chars, labels = self._load_image(index)
        while chars is None: # an error occurred, so just find another random CAPTCHA to test
            index = random.randint(0, self.__len__())
            chars, labels = self._load_image(index)
        labels = [self.char_set.index(label) for label in labels] # convert from ASCII labels to integer labels that the model uses

        chars = [torch.Tensor(char) for char in chars] # convert character images to tensors
        return {'imgs': chars, 'labels': labels}

    def __len__(self):
        return len(self.image_list)

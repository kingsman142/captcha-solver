import os
import cv2
import torch
import random
import glob
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms

class CharactersDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, validate):
        super(CharactersDataset, self).__init__()

        self.data_root = data_root
        self.validate = validate

        image_search_path = os.path.join(data_root, "**", "*.jpg")
        self.image_list = glob.glob(image_search_path, recursive = True)

        letter_set = [chr(ascii_val) for ascii_val in range(ord('A'), ord('Z') + 1)]
        number_set = [chr(ascii_val) for ascii_val in range(ord('0'), ord('9') + 1)]
        self.char_set = letter_set + number_set
        self.num_classes = len(self.char_set)

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

    def _onehot_label_encode(self, label):
        vec = np.zeros((self.num_classes))
        class_index = self.char_set.index(label)
        vec[class_index] = 1
        return vec

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
    def __init__(self, data_root, img_format, labels_root, labels_fn):
        super(CharactersDataset, self).__init__()

        self.data_root = data_root
        self.img_format = img_format
        self.labels_path = os.path.join(labels_root, labels_fn)

        self._load_labels()

    def _load_labels(self):
        with open(self.labels_path, "r") as labels_file:
            self.labels = labels_file.read().splitlines() # readlines() will include the \n character

    def _load_image(self, index):
        img_fn = self.img_format.format(self.labels[index]) # take label at index X (e.g. 'A7X6') and get filename 'A7X6.jpg'
        img_path = os.path.join(self.data_root, img_fn)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        return img

    def __getitem__(self, index):
        img = self._load_image(index)
        while img is None:
            index = random.randint(0, len(self.labels))
            img = self._load_image(index)

        return {'img': img, 'label': self.labels[index]}

    def __len__(self):
        return len(self.labels)

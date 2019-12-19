import os
import cv2
import torch
import random

class CAPTCHADataset(torch.utils.data.Dataset):
    def __init__(self, data_root, img_format, labels_root, labels_fn):
        super(CAPTCHADataset, self).__init__()

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

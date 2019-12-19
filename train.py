import torch
import torch.nn as nn
import torchvision.models as models

class DigitsClassifier(nn.Module):
    def __init__(self, num_classes = 36, pretrained = False):
        super(DigitsClassifier, self).__init__()

        self.num_classes = num_classes # default is 26 Uppercase letters + 10 digits = 36 classes

        self.model = models.alexnet(pretrained = pretrained)

    def forward(self, img):
        scores = self.model(img)
        return scores

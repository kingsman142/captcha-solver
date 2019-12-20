import torch
import torch.nn as nn
import torchvision.models as models

class CharacterClassifier(nn.Module):
    def __init__(self, num_classes = 36, pretrained = False):
        super(CharacterClassifier, self).__init__()

        self.num_classes = num_classes # default is 26 Uppercase letters + 10 digits = 36 classes

        self.model = models.alexnet(pretrained = pretrained)

        self.model.features[0] = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (11, 11), stride = (4, 4), padding = (2, 2))
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 4096, out_features = num_classes, bias = True),
            nn.Softmax()
        )

    def forward(self, img):
        scores = self.model(img)
        return scores

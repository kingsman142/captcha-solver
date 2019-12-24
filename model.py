import torch
import torch.nn as nn
import torchvision.models as models

class CharacterClassifier(nn.Module):
    def __init__(self, num_classes = 36, pretrained = False):
        super(CharacterClassifier, self).__init__()

        self.num_classes = num_classes # default is 26 Uppercase letters + 10 digits = 36 classes

        '''self.model = models.alexnet(pretrained = pretrained)

        self.model.features[0] = nn.Conv2d(in_channels = 1, out_channels = 64, kernel_size = (11, 11), stride = (4, 4), padding = (2, 2))
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 4096, out_features = num_classes, bias = True)
        )'''

        '''self.model = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = (3, 3), stride = (2, 2), padding = (2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False),
            nn.Conv2d(in_channels = 20, out_channels = 50, kernel_size = (3, 3), stride = (2, 2), padding = (2, 2)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False),
            nn.Flatten(),
            nn.Linear(in_features = 1250, out_features = 500, bias = True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 500, out_features = self.num_classes, bias = True)
        )'''

        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 1, out_channels = 20, kernel_size = (5, 5), stride = (1, 1), padding = (4, 4)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False),
            nn.Conv2d(in_channels = 20, out_channels = 50, kernel_size = (5, 5), stride = (1, 1), padding = (4, 4)),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2, padding = 0, dilation = 1, ceil_mode = False),
            nn.Flatten(),
            nn.Linear(in_features = 1250, out_features = 500, bias = True),
            nn.ReLU(inplace = True),
            nn.Linear(in_features = 500, out_features = self.num_classes, bias = True)
        )

    def forward(self, img):
        scores = self.model(img)
        return scores

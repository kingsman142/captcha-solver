import os
import torch
import torch.nn as nn

from datasets import CharactersDataset, CAPTCHADataset
from model import CharacterClassifier
from test_utils import *

# are we using GPU or CPU?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Cuda available? {}".format(torch.cuda.is_available()))
print("Device: {}".format(device))

# initialize model and load trained weights
model = CharacterClassifier(num_classes = 36, pretrained = False).to(device)
model.load_state_dict(torch.load(os.path.join(args.model_root_dir, args.model_name)))

if args.is_captcha_data: # test the model (on full 4-character CAPTCHAs)
    # set up dataset, loader, and loss function
    test_dataset = CAPTCHADataset(data_root = os.path.join("data", "captchas"))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 1)
    print("Test dataset size: {}\n".format(len(test_dataset)))

    # iterate over the CAPTCHA dataset
    captchas_correct = 0
    for batch_id, samples in enumerate(test_loader):
        labels = samples['labels'].to(device)

        imgs = samples['imgs']
        chars_incorrect = 0
        for index, img in enumerate(imgs):
            img = img.to(device)
            pred = model(img).to(device)
            if not pred == labels[index]:
                chars_incorrect += 1

        captchas_correct += 1 if (chars_incorrect == 0) else 0

        if batch_id % 1 == 0:
            print("(test) => Batch {}/{} - Current correct: {}, Running accuracy: {}%".format(batch_id, len(test_loader), captchas_correct, round(captchas_correct / batch_id * 100.0, 2)))
    test_accuracy = captchas_correct / len(test_dataset)
    print("\nCAPTCHA Test accuracy: {}%".format(round(test_accuracy * 100.0, 4)))
else: # test the model (separated characters only)
    # set up dataset, loader, and loss function
    test_dataset = CharactersDataset(data_root = os.path.join("data", "characters", "test"), validate = True)
    loss_func = nn.CrossEntropyLoss(reduction = 'sum').to(device)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = args.batch_size, shuffle = False, num_workers = 1)
    print("Test dataset size: {}\n".format(len(test_dataset)))

    # iterate over the characters dataset (contains 36 classes)
    test_loss = 0.0
    correct = 0
    for batch_id, samples in enumerate(test_loader):
        labels = samples['labels'].to(device)

        imgs = samples['imgs'].to(device)
        pred = model(imgs).to(device)

        pred_score, pred_label = torch.max(pred, 0)
        correct += 1 if pred_label == labels[0] else 0

        loss = loss_func(pred, labels)
        test_loss += loss.item()

        if batch_id % 1 == 0:
            print("(test) => Batch {}/{} - Loss: {}, Running loss: {}".format(batch_id, len(test_loader), loss.item(), test_loss))
    test_accuracy = correct / len(test_dataset)
    print("\nCharacters Test accuracy: {}%".format(round(test_accuracy * 100.0, 4)))

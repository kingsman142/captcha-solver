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
model.to(device)

if args.is_captcha_data: # test the model (on full 4-character CAPTCHAs)
    # set up dataset, loader, and loss function
    test_dataset = CAPTCHADataset(data_root = os.path.join("data", "captchas"))
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 1)
    print("Test dataset size: {}\n".format(len(test_dataset)))

    # keep track of the letters so we can do class number to letter mappings (e.g. 0 = A, 1 = B, etc.)
    letter_set = [chr(ascii_val) for ascii_val in range(ord('A'), ord('Z') + 1)]
    number_set = [chr(ascii_val) for ascii_val in range(ord('0'), ord('9') + 1)]
    char_set = letter_set + number_set

    # iterate over the CAPTCHA dataset
    captchas_correct = 0
    incorrect_letter_counts = {}
    incorrect_letters = 0
    for batch_id, samples in enumerate(test_loader):
        labels = samples['labels']

        imgs = samples['imgs']
        chars_incorrect = 0
        for index, img in enumerate(imgs):
            img = img[None, :, :, :].to(device)
            img_label = labels[index].item()
            pred = model(img) # predict score for each class
            pred = torch.argmax(pred).item() # get the index of the class with the highest score
            if not pred == img_label:
                chars_incorrect += 1
                if img_label not in incorrect_letter_counts:
                    incorrect_letter_counts[img_label] = 1
                else:
                    incorrect_letter_counts[img_label] += 1
                incorrect_letters += 1

        captchas_correct += 1 if (chars_incorrect == 0) else 0

        if batch_id % 200 == 0:
            print("(test) => Batch {}/{} - Current correct: {}, Running accuracy: {}%".format(batch_id+1, len(test_loader), captchas_correct, round(captchas_correct / (batch_id+1) * 100.0, 2)))
    test_accuracy = captchas_correct / len(test_dataset)

    for label in incorrect_letter_counts.keys():
        percent_incorrect = incorrect_letter_counts[label] / incorrect_letters # accuracy for this individual character
        char = char_set[label]
        print("{} : percent of incorrect letters = {}% ({}/{})".format(char, round(percent_incorrect * 100.0, 2), incorrect_letter_counts[label], incorrect_letters))

    print("\nCAPTCHA Test accuracy: {}%".format(round(test_accuracy * 100.0, 4)))
else: # test the model (separated characters only)
    # set up dataset, loader, and loss function
    test_dataset = CharactersDataset(data_root = os.path.join("data", "characters", "test"), validate = True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 1)
    print("Test dataset size: {}\n".format(len(test_dataset)))

    # keep track of the letters so we can do class number to letter mappings (e.g. 0 = A, 1 = B, etc.)
    letter_set = [chr(ascii_val) for ascii_val in range(ord('A'), ord('Z') + 1)]
    number_set = [chr(ascii_val) for ascii_val in range(ord('0'), ord('9') + 1)]
    char_set = letter_set + number_set

    # we use this two dictionaries to find accuracy of each character, rather than accuracy across all characters which might be subject to class imbalanace bias
    char_totals = {} # total number of times we've seen a given character
    char_correct = {} # total number of times we've gotten a character correct

    # iterate over the characters dataset (contains 36 classes)
    correct = 0
    for batch_id, samples in enumerate(test_loader):
        label = samples['labels'][0].item()

        imgs = samples['imgs'].to(device)
        pred = model(imgs).to(device)

        pred_label = torch.argmax(pred).item()
        correct += 1 if pred_label == label else 0

        if label not in char_totals:
            char_totals[label] = 1
            char_correct[label] = 1 if pred_label == label else 0
        else:
            char_totals[label] += 1
            char_correct[label] += 1 if pred_label == label else 0

        if batch_id % 100 == 0:
            print("(test) => Batch {}/{} - Current correct: {}, Running accuracy: {}%".format(batch_id+1, len(test_loader), correct, round(correct / (batch_id+1) * 100.0, 2)))
    test_accuracy = correct / len(test_dataset)

    # print accuracy for each individual character
    char_accs = [] # store the accuracies for each character here so we can sort them later on and present the accuracies in a nice ascending order
    for label in char_totals.keys():
        char_acc = char_correct[label] / char_totals[label] # accuracy for this individual character
        char_accs.append((char_acc, char_correct[label], char_totals[label], char_set[label])) # append character accuracy, as well as the character itself, to the accuracy list
    char_accs.sort(key = lambda x : x[0])
    for item in char_accs: # print accuracies for each character in descending order
        acc, correct, total, char = item
        print("{} : {}% ({}/{})".format(char, round(acc * 100.0, 2), correct, total))

    # print overall accuracy across all characters
    print("\nCharacters Test accuracy: {}%".format(round(test_accuracy * 100.0, 4)))

import os
import glob
import shutil
import sklearn.model_selection as skms

def move_images_to_split(paths, split_identifier, char_set): # split = train, test, val
    # create data/characters/train, data/characters/test, and data/characters/val
    split_path = os.path.join("data", "characters", split_identifier)
    if not os.path.exists(split_path):
        os.mkdir(split_path)

    # e.g. create data/characters/train/A, data/characters/train/B, etc...
    for char in char_set:
        split_char_path = os.path.join("data", "characters", split_identifier, char)
        if not os.path.exists(split_char_path):
            os.mkdir(split_char_path)

    # move all character images into their respective split directory
    for path in paths:
        img_fn = os.path.split(path)[1] # e.g. "A_1583.jpg"
        img_label = img_fn.split(".")[0].split("_")[0] # e.g. A, B, C, ..., 1, 2, 3, ...
        output_path = os.path.join("data", "characters", split_identifier, img_label, img_fn)
        shutil.copyfile(path, output_path)

# find all characters used in the classification task (26 letters + 10 digits = 36 total)
letter_set = [chr(ascii_val) for ascii_val in range(ord('A'), ord('Z') + 1)]
number_set = [chr(ascii_val) for ascii_val in range(ord('0'), ord('9') + 1)]
char_set = letter_set + number_set

# find paths to all extracted digits
characters_search_string = os.path.join("data", "characters", "**", "*.jpg")
character_paths = glob.glob(characters_search_string, recursive = True)

TRAIN_SIZE = int(0.8 * len(character_paths))
TEST_SIZE = int(0.2 * len(character_paths))
VAL_SIZE = int(0.2 * len(character_paths))

train_paths, test_paths = skms.train_test_split(character_paths, train_size = TRAIN_SIZE)
train_paths, val_paths = skms.train_test_split(train_paths, test_size = VAL_SIZE)

move_images_to_split(train_paths, "train", char_set)
move_images_to_split(val_paths, "val", char_set)
move_images_to_split(test_paths, "test", char_set)

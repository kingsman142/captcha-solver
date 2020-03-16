This repository aims to create a CAPTCHA solver for basic text CAPTCHAs.  It currently uses the captcha pip library located here: https://github.com/lepture/captcha/ .

Other CAPTCHA-generating libraries worth looking into are https://github.com/kuszaj/claptcha (CLAPTCHA) and https://www.dev2qa.com/how-to-generate-random-captcha-in-python/ (Tutorial).

How to run:
1) Run `pip3 install -r requirements.txt`
2) Run `python3 generate_captchas.py N` where N is the number of CAPTCHAs to generate
3) (optional) Run `python3 visualize_images.py N` to scroll through N CAPTCHAs (N is not a required command-line argument; default is 20)
4) Run `python3 preprocess_dataset.py` to preprocess the dataset and split out the characters from the CAPTCHAs
5) Run `python3 make_train_val_test_split.py` to split individual segmented characters into train/val/test splits
6) Run `python3 train.py --num-epochs 50 --learning-rate 0.00005 --weight-decay 0.98 --batch-size 512` to train the model on the characters
7) (optional) Run `python3 test.py --captchas` to get accuracy on test dataset of CAPTCHAs
8) (optional) Run `python3 test.py --characters` to get accuracy on test dataset of individual characters (should be a lot higher than CAPTCHAs accuracy)
9) (optional) Run `python3 visualize_classes.py` to visualize TSNE plots of characters before training and after training

A well-thought-out report detailing the development of this project can be found on Medium, here: https://medium.com/@jameshahn_27452/solving-noisy-text-captchas-126734c3c717 .

A pre-trained model can be found at https://drive.google.com/open?id=16Vwha7uxy7coe9y-Nkh6skYPW3Kz8xZA . In order to use the model, please download it, create a "models/" directory in this project's root directory, and then place the pre-trained model into the models/ folder. Then, skip to step 5 above.

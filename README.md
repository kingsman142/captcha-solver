This repository aims to create a CAPTCHA solver for basic text CAPTCHAs.  It currently uses the captcha pip library located here: https://github.com/lepture/captcha/ .

Other CAPTCHA-generating libraries worth looking into are https://github.com/kuszaj/claptcha (CLAPTCHA) and https://www.dev2qa.com/how-to-generate-random-captcha-in-python/ (Tutorial).

How to run:
0) Run `pip3 install -r requirements.txt`
1) Run `python3 generate_captchas.py N` where N is the number of CAPTCHAs to generate
2) Run `python3 preprocess_dataset.py` to preprocess the dataset and split out the characters from the CAPTCHAs
3) Run `python3 make_train_val_test_split.py` to split individual segmented characters into train/val/test splits
4) Run `python3 train.py` to train the model on the characters
5) (optional) Run `python3 test.py --captchas` to get accuracy on test dataset of CAPTCHAs
6) (optional) Run `python3 test.py --characters` to get accuracy on test dataset of individual characters (should be a lot higher than CAPTCHAs accuracy)
7) (optional) Run `python3 visualize_classes.py` to visualize TSNE plots of characters before training and after training

Challenges:
1) Characters are not fixed-width in size
2) Characters are not fixed-height in size
3) Characters are not in fixed x or y coordinates
4) Characters are not at a fixed angle
5) Images vary in color, and come in RGB instead of grayscale
6) Image has circular noise
7) Circular noise is not like salt-and-pepper noise where it's just one pixel that can be removed with a median filter
8) Image has line noise
9) Line noise is not perfectly linear, but rather curvilinear, so typical gradient techniques don't work
10) Characters are not all at the same scale (some larger/smaller than others)
11) When image is read in as grayscale, since noise is not always lighter or always darker than characters, a thresholding technique can't be used to denoise image

Things I tried:
1) Gaussian blue image to turn rectangle noise into circular noise, then hough transform to detect circles and remove them
2) Vertical median filter to remove horizontal line noise (coincidentally also removed the circle noise
3) Erosion + median filter, then dilation
4) Segment characters by column minima rule (count number of character pixels per column and segment on each column minima)
5) If there are conflicts in character segmentations, take the confusing region and take the minimum (naive approach)

Assumptions in approach:
1) (preprocessing) line noise is always the same width in all captchas
2) (preprocessing) line noise is always thinner than character stroke width
3) (preprocessing) images are always 140x76
4) (preprocessing) circle noise is always same radius
5) (preprocessing) only 4 characters used in CAPTCHA
6) (preprocessing) CAPTCHA dataset images all look roughly the same style, and the style isn't the most complex version out there
7) (preprocessing) line noise is always horizontal in general, and never vertical
8) (dataset) characters are not hollowed out
9) (dataset) CAPTCHA is not 3D
10) (dataset) no lowercase letters are used
11) (dataset) at most 2 characters are conjoined at once
12) (preprocessing) there is enough of a contrast between the background and lettering that thresholding is effective

Ablation studies:
1) does threshold help?
2) does denoising help?
3) does splitting conjoined characters by the middle and by minima change performance?
4) does white padding of the character matter on the left and right side of the character?
5) what would happen if we balanced the digits dataset?

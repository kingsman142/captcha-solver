import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type = int, action = 'store', default = 64)
parser.add_argument('--model-root-dir', type = str, action = 'store', default = 'models')
parser.add_argument('--model-name', type = str, action = 'store', default = 'epoch_7_lr_5e-05_batchsize_512')
parser.add_argument('--captchas', dest = 'is_captcha_data', action = 'store_true')
parser.add_argument('--characters', dest = 'is_captcha_data', action = 'store_false')
parser.set_defaults(is_captcha_data = True)
args = parser.parse_args()

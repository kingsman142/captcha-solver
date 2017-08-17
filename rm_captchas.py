import os
import glob

files = glob.glob("./captchas/*")
for f in files:
    os.remove(f)

#!/usr/bin/env python

import os
import subprocess
import glob
import numpy as np
from PIL import Image


def main():
    path = '../image_data/beautiful/food/'
    files = glob.glob(path + '*jpg')
    os.makedirs("./resized", exist_ok=True)

    for f_in in files[:2]:
        base_name = os.path.basename(f_in)
        img = Image.open(f_in)
        w, h = img.size

        x = np.random.randint(w - 256)
        y = np.random.randint(h - 256)

        f_out = f"./resized/{base_name}"

        cropped_image = img.crop((x, y, x + 256, y + 256))
        cropped_image.save(f_out)


if __name__ == "__main__":
    main()

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
    crop_size = 256
    n = 10

    for f_in in files:
        base_name = os.path.basename(f_in)
        tmps = base_name.split(".")

        img = Image.open(f_in)
        w, h = img.size

        if w < crop_size or h < crop_size:
            continue

        for i in range(n):
            tmp = f"{tmps[0]}_{i}.{tmps[1]}"
            x = np.random.randint(w - crop_size)
            y = np.random.randint(h - crop_size)

            f_out = f"./resized/{tmp}"

            cropped_image = img.crop((x, y, x + 256, y + 256))
            cropped_image.save(f_out)


if __name__ == "__main__":
    main()

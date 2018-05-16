#!/usr/bin/env python

import os
import subprocess
import glob
import numpy as np


def main():
    #path = '../image_data/beautiful/food/'
    path = './resized/'
    files = glob.glob(path + '*jpg')
    os.makedirs("./blured", exist_ok=True)

    for f_in in files:
        angle = np.random.randint(180)
        radius = np.random.randint(100)
        sigma = np.random.randint(100)

        base_name = os.path.basename(f_in)

        f_out = f"./blured/{base_name}"

        cmd = f"convert {f_in} -motion-blur {angle}x{radius}+{angle} {f_out}"
        print(cmd)

        subprocess.call(cmd, shell=True)

    print("Done!")


if __name__ == "__main__":
    main()

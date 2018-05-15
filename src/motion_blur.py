#!/usr/bin/env python

import os
import subprocess
import glob
import numpy as np


def main():
    path = '../image_data/beautiful/food/'
    files = glob.glob(path + '*jpg')
    os.makedirs("./blured", exist_ok=True)

    n = 10

    for f_in in files:

        for i in range(n):

            angle = np.random.randint(180)
            radius = np.random.randint(100)
            sigma = np.random.randint(100)

            base_name = os.path.basename(f_in)

            tmp = base_name.split(".")
            base_name = tmp[0] + "_" + str(i) + "." + tmp[1]

            f_out = f"./blured/{base_name}"

            cmd = f"convert {f_in} -motion-blur {angle}x{radius}+{angle} {f_out}"
            print(cmd)

            subprocess.call(cmd, shell=True)

    print("Done!")


if __name__ == "__main__":
    main()

#!/usr/bin/env python

import os
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image


class ImageConverter:

    @staticmethod
    def resize(input_dir:str, output_dir: str, crop_size = 256, n=10):
        """
        :param input_dir: An input directory path
        :param output_dir: An output directory path
        :param crop_size: Size of crop
        :param n: The number of generation for an input image
        :return: None
        """

        if output_dir == input_dir:
            raise Exception("Input directory and output directory are same.")

        p = Path(input_dir)
        files = p.glob('*.jpg')
        os.makedirs(output_dir, exist_ok=True)

        for f_in in files:
            base_name = os.path.basename(f_in)
            fname, ext = base_name.split(".")

            img = Image.open(f_in)
            w, h = img.size

            if w < crop_size or h < crop_size:
                continue

            for i in range(n):
                out_name = f"{fname}_{i}.{ext}"
                x = np.random.randint(w - crop_size)
                y = np.random.randint(h - crop_size)

                f_out = f"./{output_dir}/{out_name}"
                cropped_image = img.crop((x, y, x + crop_size, y + crop_size))
                cropped_image.save(f_out)

    @staticmethod
    def motion_bluer(input_dir: str, output_dir: str):
        """
        :param input_dir: An input directory path
        :param output_dir: An output directory path
        :return:
        """

        if output_dir == input_dir:
            raise Exception("Input directory and output directory are same.")

        p = Path(input_dir)
        files = p.glob('*.jpg')
        os.makedirs(output_dir, exist_ok=True)

        for f_in in files:
            angle = np.random.randint(180)
            radius = np.random.randint(100)

            base_name = os.path.basename(f_in)

            f_out = f"./{output_dir}/{base_name}"
            cmd = f"convert {f_in} -motion-blur {angle}x{radius}+{angle} {f_out}"
            subprocess.call(cmd, shell=True)

    @staticmethod
    def gamma_correction(input_dir: str, output_dir: str):
        import cv2
        import random

        if output_dir == input_dir:
            raise Exception("Input directory and output directory are same.")

        p = Path(input_dir)
        files = p.glob('*.jpg')
        os.makedirs(output_dir, exist_ok=True)

        lookup_table = np.zeros((256, 1), dtype='uint8')

        for f_in in files:

            gamma = random.uniform(0.1, 0.2)
            for i in range(256):
                lookup_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)

            img = cv2.imread(str(f_in))
            img_gamma = cv2.LUT(img, lookup_table)
            base_name = os.path.basename(f_in)
            f_out = f"{output_dir}/{base_name}"
            cv2.imwrite(f_out, img_gamma)


def main():
    converter = ImageConverter()
    #converter.resize('../image_data/beautiful/food/', './fuga')
    #converter.motion_bluer('./resized/', './hoge')
    converter.gamma_correction('./resized/', './dark')


if __name__ == "__main__":
    main()

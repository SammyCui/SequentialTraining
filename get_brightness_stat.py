import os
from PIL import Image,ImageStat
import sys
import numpy as np


def calculate_brightness(image):
    """
    ref: https://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python
    :param image:
    :return:
    """
    greyscale_image = image.convert('L')
    histogram = greyscale_image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * (-scale + index)

    return 1 if brightness == 255 else brightness / scale

def brightness( im_file ):
   im = im_file.convert('L')
   stat = ImageStat.Stat(im)
   return stat.mean[0]

def get_brightness_stats(root_path):
    brightness_per_cat = {}
    for cat in os.listdir(root_path):
        curr_cat = []
        for image_name in os.listdir(os.path.join(root_path,cat)):
            image = Image.open(os.path.join(root_path,cat,image_name))
            curr_cat.append(brightness(image))
        brightness_per_cat[cat] = np.mean(curr_cat)
    
    return brightness_per_cat


if __name__ == '__main__':
    root_path = sys.argv[1]
    print(get_brightness_stats(root_path))

import os

import torch
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision
import numpy as np
from typing import Tuple, Optional, Callable, Any, List, Dict, Iterable
import xmltodict
import re
import shutil
import pandas as pd
import sys
from torchvision.datasets.folder import is_image_file
from torchvision.transforms import transforms


def generate_noise(img):
    imgArr = np.array(img)  # convert img to np array
    f = np.fft.fftshift(np.fft.fftn(imgArr))  # fourier transform of image
    fPhase = np.angle(f)  # phase component of fft
    fMag = np.abs(f)  # magnitude component of fft

    rng = np.random.default_rng(1)  # rng seed
    rng.shuffle(fPhase, 0)  # shuffle phases in x
    rng.shuffle(fPhase, 1)  # shuffle phases in y

    combined = np.multiply(np.abs(f), np.exp(1j * fPhase))  # recombine magnitude with shifted phases
    imgCombined = np.real(np.fft.ifftn(combined))  # inverse fft to recreate original image
    imgCombined = np.abs(imgCombined)  # take absolute value of recombination to-
    # eliminate value clipping errors in final image

    absfImg = Image.fromarray(imgCombined.astype('uint8'), 'RGB')  # convert phase-shifted noise array to PIL Image
    # absfImg.show('cat.png')     #optional show image
    # absfImg.save('cat.png')     #optional save image

    # absfImg = ScaleRGBPowers(imgCombined, imgArr)

    return absfImg


def get_bndbox(annotation_path: str, cat: str) -> Tuple[int, int, int, int]:
    with open(annotation_path, "r") as xml_obj:
        # converting the xml data to Python dictionary
        my_dict = xmltodict.parse(xml_obj.read())
        # closing the file
        xml_obj.close()
    obj = my_dict['annotation']['object']
    if isinstance(obj, list):
        h_bb_max, w_bb_max = 0, 0
        x_min_best, y_min_best, x_max_best, y_max_best = 0, 0, 0, 0
        for img_obj in obj:
            if img_obj['name'] == cat:
                x_min = int(img_obj['bndbox']['xmin'])
                y_min = int(img_obj['bndbox']['ymin'])
                x_max = int(img_obj['bndbox']['xmax'])
                y_max = int(img_obj['bndbox']['ymax'])
                if (y_max - y_min) * (x_max - x_min) > h_bb_max * w_bb_max:
                    h_bb_max, w_bb_max = y_max - y_min, x_max - x_min
                    x_min_best = int(img_obj['bndbox']['xmin'])
                    y_min_best = int(img_obj['bndbox']['ymin'])
                    x_max_best = int(img_obj['bndbox']['xmax'])
                    y_max_best = int(img_obj['bndbox']['ymax'])
    else:
        x_min_best = int(obj['bndbox']['xmin'])
        y_min_best = int(obj['bndbox']['ymin'])
        x_max_best = int(obj['bndbox']['xmax'])
        y_max_best = int(obj['bndbox']['ymax'])

    return x_min_best, y_min_best, x_max_best, y_max_best


class GenerateBackground:
    """
    generate background on specific size
    """

    def __init__(self, bg_type: str, bg_color: Tuple[int, int, int] = None):
        """

        :param bg_type:
        :param img:
        :param bg_color:
        """
        self.bg_type = bg_type
        self.bg_color = bg_color

    def __call__(self, bg_size: Tuple[int, int], img: Image.Image = None):
        """

        :param bg_size: (H,W)
        :return:
        """
        H, W = bg_size
        if self.bg_type == 'fft':
            assert img is not None, 'FFT mode requires an input image'
            # pil use w,h instead of h,w
            if (img.size[1] != H) or (img.size[0] != W):
                img = torchvision.transforms.Resize(bg_size)(img)
            return generate_noise(img=img)
        elif self.bg_type == 'color':
            rgb = self.bg_color if self.bg_color else (169, 169, 169)
            return Image.new('RGB', (W, H), rgb)

    def __str__(self):
        return self.bg_type

    @staticmethod
    def generate_noise(img: Image.Image):
        imgArr = np.array(img)  # convert img to np array
        f = np.fft.fftshift(np.fft.fftn(imgArr))  # fourier transform of image
        fPhase = np.angle(f)  # phase component of fft
        fMag = np.abs(f)  # magnitude component of fft

        rng = np.random.default_rng(40)
        rng.shuffle(fPhase, 0)  # shuffle phases in x
        rng.shuffle(fPhase, 1)  # shuffle phases in y

        combined = np.multiply(np.abs(f), np.exp(1j * fPhase))  # recombine magnitude with shifted phases
        imgCombined = np.real(np.fft.ifftn(combined))  # inverse fft to recreate original image
        imgCombined = np.abs(imgCombined)  # take absolute value of recombination to-
        # eliminate value clipping errors in final image

        absfImg = GenerateBackground.ScaleRGBPowers(imgCombined, imgArr)

        return absfImg

    @staticmethod
    def ScaleRGBPowers(noise, src):
        # split the noise and src into their rgb components
        nr, ng, nb = np.split(noise, 3, 2)
        sr, sg, sb = np.split(src, 3, 2)

        # scale the noise rgb components to match the sumsquared value of the src image
        rr = nr / (np.sqrt(np.sum(nr ** 2) / np.sum(sr ** 2)))
        gg = ng / (np.sqrt(np.sum(ng ** 2) / np.sum(sg ** 2)))
        bb = nb / (np.sqrt(np.sum(nb ** 2) / np.sum(sb ** 2)))

        # recombine the scaled rgb components
        rgb = np.dstack((rr, gg, bb))

        # convert the noise from np array to PIL Image
        newImg = Image.fromarray((rgb).astype('uint8'), 'RGB')
        # newImg.show()      #optional image show

        return newImg


def check_bb_size(anno_root: str):
    h_w_ratio = {}
    n_pixels = {}
    for cat in os.listdir(anno_root):
        if cat not in h_w_ratio:
            h_w_ratio[cat] = []
        if cat not in n_pixels:
            n_pixels[cat] = []
        for image_name in os.listdir(os.path.join(anno_root, cat)):
            annotation_path = os.path.join(anno_root, cat, image_name)
            with open(annotation_path, "r") as xml_obj:
                # coverting the xml data to Python dictionary
                my_dict = xmltodict.parse(xml_obj.read())
                # closing the file
                xml_obj.close()

            obj = my_dict['annotation']['object']
            if isinstance(obj, list):
                h_bb_max, w_bb_max = 0, 0
                for object in obj:
                    if object['name'] == cat:
                        x_min = int(object['bndbox']['xmin'])
                        y_min = int(object['bndbox']['ymin'])
                        x_max = int(object['bndbox']['xmax'])
                        y_max = int(object['bndbox']['ymax'])
                        if (y_max - y_min) * (x_max - x_min) > h_bb_max * w_bb_max:
                            h_bb_max, w_bb_max = y_max - y_min, x_max - x_min
            else:
                x_min = int(obj['bndbox']['xmin'])
                y_min = int(obj['bndbox']['ymin'])
                x_max = int(obj['bndbox']['xmax'])
                y_max = int(obj['bndbox']['ymax'])
                h_bb_max, w_bb_max = y_max - y_min, x_max - x_min

            h_w_ratio[cat].append(h_bb_max / w_bb_max)
            n_pixels[cat].append(h_bb_max * w_bb_max)
    return h_w_ratio, n_pixels


def generate_test_set(anno_root: str, image_root: str, test_anno_root: str, test_image_root: str, size: float,
                      random_state: int = 40):
    """
    randomly move portion of the images and annos from current dir to another
    :param anno_root:
    :param image_root:
    :param test_anno_root:
    :param test_image_root:
    :param size:
    :param random_state:
    :return:
    """
    img_arr, label_arr = [], []
    for cat in os.listdir(image_root):
        for image in os.listdir(os.path.join(image_root, cat)):
            image_name = image.split('.')[0]
            img_arr.append(image_name)
            label_arr.append(cat)

    X_train, X_test, y_train, y_test = train_test_split(img_arr, label_arr, test_size=size, random_state=random_state)
    for image_name, image_label in zip(X_test, y_test):
        if not os.path.isdir(os.path.join(test_image_root, image_label)):
            os.mkdir(os.path.join(test_image_root, image_label))
        if not os.path.isdir(os.path.join(test_anno_root, image_label)):
            os.mkdir(os.path.join(test_anno_root, image_label))
        shutil.move(os.path.join(image_root, image_label, image_name) + '.jpg',
                    os.path.join(test_image_root, image_label, image_name) + '.jpg')
        shutil.move(os.path.join(anno_root, image_label, image_name) + '.xml',
                    os.path.join(test_anno_root, image_label, image_name) + '.xml')


def is_bndbox_big(anno_path: str, threshold):
    cat = anno_path.split('/')[-2]
    with open(anno_path, "r") as xml_obj:
        # coverting the xml data to Python dictionary
        my_dict = xmltodict.parse(xml_obj.read())
        # closing the file
        xml_obj.close()

    obj = my_dict['annotation']['object']
    if isinstance(obj, list):
        h_bb_max, w_bb_max = 0, 0
        for object in obj:
            if object['name'] == cat:
                x_min = int(object['bndbox']['xmin'])
                y_min = int(object['bndbox']['ymin'])
                x_max = int(object['bndbox']['xmax'])
                y_max = int(object['bndbox']['ymax'])
                if (y_max - y_min) * (x_max - x_min) > h_bb_max * w_bb_max:
                    h_bb_max, w_bb_max = y_max - y_min, x_max - x_min
    else:
        x_min = int(obj['bndbox']['xmin'])
        y_min = int(obj['bndbox']['ymin'])
        x_max = int(obj['bndbox']['xmax'])
        y_max = int(obj['bndbox']['ymax'])
        h_bb_max, w_bb_max = y_max - y_min, x_max - x_min

    return (h_bb_max > threshold or w_bb_max > threshold) and (h_bb_max > 0) and (w_bb_max > 0)


def filter_small_bndbox(anno_root: str, threshold=150):
    """
    delete image-anno pairs with bndbox's longer side shorter than threshold
    :param anno_root:
    :param image_root:
    :param new_anno_root:
    :param new_image_root:
    :param threshold:
    :return:
    """

    not_ok_dict = {}
    for cat in os.listdir(anno_root):
        if 'tar.gz' in cat:
            continue
        not_ok_dict[cat] = 0
        # if not os.path.isdir(os.path.join(new_anno_root, cat)):
        #     os.mkdir(os.path.join(new_anno_root, cat))
        # if not os.path.isdir(os.path.join(new_image_root, cat)):
        #     os.mkdir(os.path.join(new_image_root, cat))
        cur_root = os.path.join(anno_root, cat, 'Annotation', cat)
        for image_name in os.listdir(cur_root):
            annotation_path = os.path.join(cur_root, image_name)

            # if h_bb_max > threshold or w_bb_max > threshold:
            #     image_name = image_name.split('.')[0]
            #     shutil.copyfile(os.path.join(image_root, cat, image_name) + '.jpg',
            #                     os.path.join(new_image_root, cat, image_name) + '.jpg')
            #     shutil.copyfile(os.path.join(anno_root, cat, image_name) + '.xml',
            #                     os.path.join(new_anno_root, cat, image_name) + '.xml')

            if not is_bndbox_big(annotation_path, threshold):
                print(image_name, ' < 150')
                not_ok_dict[cat] += 1
            else:
                print()
    print(not_ok_dict)


class IsValidFileImagenet:
    """
    ref: torch is_valid_file. Callable object for checking if an image file is valid.

    """

    def __init__(self, anno_root, threshold: int = 150):
        self.anno_root = anno_root
        self.threshold = threshold

    def __call__(self, path: str):
        image_name = path.split('/')[-1].split('.')[0]
        image_cat = image_name.split('_')[0]
        xml_path = os.path.join(self.anno_root, image_cat, 'Annotation', image_cat, image_name + '.xml')
        return is_image_file(path) and os.path.isfile(xml_path) and is_bndbox_big(xml_path, threshold=self.threshold)


def sanity_check(root: str, anno: str):
    """
    check if image is valid. Delete both img and anno if not
    :param root: image
    :param anno: anotations
    :return:
    """

    for cat in os.listdir(root):
        for img in os.listdir(os.path.join(root, cat)):
            name = img.split(".")[0]
            try:
                Image.open(os.path.join(root, cat, img))
            except:
                print(f"{os.path.join(root, cat, img)} is not available.")
                os.remove(os.path.join(root, cat, img))
                os.remove(os.path.join(anno, cat, name + '.xml'))


def dataset_dimensions(anno: str):
    """
    check if image is valid. Delete both img and anno if not
    :param root: image
    :param anno: anotations
    :return:
    """
    stats = pd.DataFrame(columns=['Count', 'H_avg', 'W_avg'])
    for cat in os.listdir(anno):
        h_avg = 0
        w_avg = 0
        count = 0
        for ann in os.listdir(os.path.join(anno, cat)):
            with open(os.path.join(anno, cat, ann), "r") as xml_obj:
                # coverting the xml data to Python dictionary
                my_dict = xmltodict.parse(xml_obj.read())
                # closing the file
                xml_obj.close()
            h = my_dict['annotation']['height']
            w = my_dict['annotation']['width']
            h_avg += h
            w_avg += w
            count += 1
        h_avg /= count
        w_avg /= count

        row = [count, h_avg, w_avg]
        stats.loc[cat] = row
    return stats


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


def get_brightness_stats(root_path):
    brightness_per_cat = {}
    for cat in os.listdir(root_path):
        curr_cat = []
        for image_name in os.listdir(os.path.join(root_path, cat)):
            image = Image.open(os.path.join(root_path, cat, image_name))
            curr_cat.append(calculate_brightness(image))
        brightness_per_cat[cat] = np.mean(curr_cat)

    return brightness_per_cat


def remove_bad_images(image_root: str):
    for cat in os.listdir(image_root):
        print(f'==>Scanning {cat}')
        for image in os.listdir(os.path.join(image_root, cat)):
            try:
                if image.endswith(('.JPEG', '.jpg', '.png')):
                    img = Image.open(os.path.join(image_root,cat,image))
            except UserWarning:
                print('Corrupt EXIF data on file: ', os.path.join(image_root,cat,image))
            except Exception:
                print('Bad image: ', os.path.join(image_root,cat,image))


if __name__ == '__main__':
    args = sys.argv
    globals()[args[1]](*args[2:])

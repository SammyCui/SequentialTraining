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
import warnings

warnings.filterwarnings('error')

from torch_dataset_06 import ResizingCIFAR10


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


class VOCDistancingImageLoader:
    """
        simgle data loader for torch datasets. Full doc: https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
        only work for VOC file structure
    """

    def __init__(self, size: Tuple[int, int], p: float, annotation_root_path: str,
                 background_generator: Optional[Callable] = None,
                 dataset_name: str = 'VOC',
                 resize_method: str = 'long',
                 draw_bndbox_width: int = None):
        """

        :param size: size of the full final training image (bounding box + rest-of-image + background)
        :param p: H_bb / size[0] or W_bb / size[1], depending on what resize_method is used
        :param background: string specify which
        :param annotation_root_path:
        :param resize_method: if 'long', resize bounding box to p=max(H_bb,W_bb)/corresponding H/W of size
                              if 'short',                       p=min(H_bb,W_bb)/corresponding H/W of size with center crop
                              TODO: if resize_method='short', consider cropping on two ends of the longer side
        """
        self.H_final, self.W_final = size
        self.p = p
        self.background_generator = background_generator
        self.annotation_root_path = annotation_root_path
        self.resize_method = resize_method
        self.dataset_name = dataset_name
        self.draw_bndbox_width = draw_bndbox_width

    def __call__(self, path: str):
        img = Image.open(path)
        img = img.convert("RGB")

        path_component = re.split(r'\.|/', path)
        cat, name = path_component[-3], path_component[-2]
        if self.dataset_name == 'imagenet':
            annotation_path = os.path.join(self.annotation_root_path, cat, 'Annotation', cat, name + '.xml')
        else:
            annotation_path = os.path.join(self.annotation_root_path, cat, name + '.xml')

        x_min, y_min, x_max, y_max = get_bndbox(annotation_path=annotation_path, cat=cat)
        if self.resize_method == 'adjust':
            bnd_box_img = img.crop((x_min, y_min, x_max, y_max))
            h_resized, w_resized = int(self.H_final * self.p), int(self.W_final * self.p)
            resizer = torchvision.transforms.Resize((h_resized, w_resized))
            return resizer(bnd_box_img)

        elif self.resize_method == 'long':

            H, W = img.size[1], img.size[0]
            background = self.background_generator((self.H_final, self.W_final), img)

            H_bb, W_bb = y_max - y_min, x_max - x_min
            # longer_side = 'H_bb' if H_bb > W_bb else 'W_bb'

            if H_bb > W_bb:
                # get the size of resized bndbox
                H_bb_target = self.H_final * self.p
                assert H_bb_target <= H_bb, 'This bndbox longest side is smaller than the target side. Cannot resize the target bigger'
                W_bb_target = H_bb_target * W_bb / H_bb
            else:
                # get the size of resized bndbox
                W_bb_target = self.W_final * self.p
                assert W_bb_target <= W_bb, 'This bndbox longest side is smaller than the target side. Cannot resize the target bigger'
                H_bb_target = W_bb_target * H_bb / W_bb

            # get the size of orig image after resizing bndbox
            H_target = int(np.floor(H_bb_target * H / H_bb))
            W_target = int(np.floor(W_bb_target * W / W_bb))
            # resize original image
            resized_img = torchvision.transforms.Resize((H_target, W_target))(img)
            resized_img_array = np.array(resized_img)
            # get coordinates of resized bndbox in the resized image
            x_min_resized, x_max_resized = int(np.floor(x_min / W * W_target)), int(np.floor(x_max / W * W_target))
            y_min_resized, y_max_resized = int(np.floor(y_min / H * H_target)), int(np.floor(y_max / H * H_target))

            # get coordinates in the resized image to crop (bounding box plus the surroundings)
            # from the resized orig image (to place in the final image)

            # if it goes out of the left bound of the resized image, let it be 0
            x_min_orig = max(0, int(np.floor(x_min_resized - (self.W_final - W_bb_target) / 2)))
            # if it goes out of the right bound, let it be the right bound
            x_max_orig = min(W_target, int(np.floor(x_max_resized + (self.W_final - W_bb_target) / 2)))

            # if it goes out of the top bound of the resized image, let it be 0
            y_min_orig = max(0, int(np.floor(y_min_resized - (self.H_final - H_bb_target) / 2)))
            # if it goes out of the bottom bound, let it be the bottom bound
            y_max_orig = min(H_target, int(np.floor(y_max_resized + (self.H_final - H_bb_target) / 2)))
            if y_max_orig - y_min_orig - self.H_final > 1:
                raise Exception("Sth. went wrong during resizing for path: ", path)
            elif 0 < y_max_orig - y_min_orig - self.H_final <= 1:
                y_max_orig -= y_max_orig - y_min_orig - self.H_final
            if x_max_orig - x_min_orig - self.W_final > 1:
                raise Exception("Sth. went wrong during resizing for path: ", path)
            elif 0 < x_max_orig - x_min_orig - self.W_final <= 1:
                x_max_orig -= x_max_orig - x_min_orig - self.W_final
            # get coordinates to place the above cropped resized image in the final image

            # 1. get coordinates of resized bndbox in the final image
            x_min_final_bb, x_max_final_bb = int(np.floor((self.W_final - W_bb_target) / 2)), int(
                np.round(self.W_final / 2 + W_bb_target / 2))
            y_min_final_bb, y_max_final_bb = int(np.floor((self.H_final - H_bb_target) / 2)), int(
                np.round(self.H_final / 2 + H_bb_target / 2))

            # 2. Add/subtract surrounding distances of resized-bndbox-in-resized-orig-image to
            # each coordinates of resized bndbox in the final image

            x_min_final, y_min_final = max(0, x_min_final_bb - (x_min_resized - x_min_orig)), max(0, y_min_final_bb - (
                    y_min_resized - y_min_orig))
            # if x_max_orig - x_min_orig > W_target:
            #     x_min_orig += 1
            # if y_max_orig - y_min_orig > H_target:
            #     y_min_orig += 1

            x_max_final, y_max_final = x_min_final + x_max_orig - x_min_orig, y_min_final + y_max_orig - y_min_orig
            if x_max_final - 1 == self.W_final:
                x_min_final, x_max_final = x_min_final - 1, x_max_final - 1
            if y_max_final - 1 == self.W_final:
                y_min_final, y_max_final = y_min_final - 1, y_max_final - 1

            # place the resized-orig-image onto the final image, which has only background right now

            final_img_array = np.array(background)

            try:
                final_img_array[int(np.floor(y_min_final)):int(np.floor(y_max_final)),
                int(np.floor(x_min_final)):int(np.floor(x_max_final))] = \
                    resized_img_array[int(np.floor(y_min_orig)):int(np.floor(y_max_orig)),
                    int(np.floor(x_min_orig)):int(np.floor(x_max_orig))]

                if self.draw_bndbox_width:
                    final_img_array[y_min_final_bb:y_min_final_bb + self.draw_bndbox_width,
                    x_min_final_bb:x_max_final_bb] = np.full(
                        (self.draw_bndbox_width, x_max_final_bb - x_min_final_bb, 3), 255, dtype='uint8')
                    final_img_array[y_max_final_bb - self.draw_bndbox_width:y_max_final_bb,
                    x_min_final_bb:x_max_final_bb] = np.full(
                        (self.draw_bndbox_width, x_max_final_bb - x_min_final_bb, 3), 255, dtype='uint8')
                    final_img_array[y_min_final_bb:y_max_final_bb,
                    x_min_final_bb:x_min_final_bb + self.draw_bndbox_width] = np.full(
                        (y_max_final_bb - y_min_final_bb, self.draw_bndbox_width, 3), 255, dtype='uint8')
                    final_img_array[y_min_final_bb:y_max_final_bb,
                    x_max_final_bb - self.draw_bndbox_width:x_max_final_bb] = np.full(
                        (y_max_final_bb - y_min_final_bb, self.draw_bndbox_width, 3), 255, dtype='uint8')

            except:
                raise ValueError(f"size doesn't match. path: {path}")

            return Image.fromarray(final_img_array)

    def crop(self):
        pass


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


class NoAnnotationImageLoader:
    def __init__(self,
                 size: Tuple[int, int],
                 p: float,
                 background_generator: Callable,
                 resize_method: str = 'long'
                 ):

        self.H_final, self.W_final = size
        self.p = p
        self.background_generator = background_generator
        self.resize_method = resize_method

    def __call__(self, path: str):
        img = Image.open(path)
        img = img.convert("RGB")

        if self.resize_method == 'adjust':

            h_resized, w_resized = int(self.H_final * self.p), int(self.W_final * self.p)
            resizer = torchvision.transforms.Resize((h_resized, w_resized))
            return resizer(img)

        elif self.resize_method == 'long':

            H, W = img.size[1], img.size[0]
            if H > W:
                # get the size of resized bndbox
                H_target = self.H_final * self.p
                assert H_target <= H, 'This bndbox longest side is smaller than the target side. Cannot resize the target bigger'
                W_target = H_target * W / H
            else:
                # get the size of resized bndbox
                W_target = self.W_final * self.p
                assert W_target <= W, 'This bndbox longest side is smaller than the target side. Cannot resize the target bigger'
                H_target = W_target * H / W
            H_target, W_target = int(H_target), int(W_target)
            background = self.background_generator((self.H_final, self.W_final))
            resized_img = torchvision.transforms.Resize((H_target, W_target))(img)
            resized_img_array = np.array(resized_img)
            final_img_array = np.array(background)
            x_left, y_top = int((self.W_final - W_target) / 2), int((self.H_final - H_target) / 2)
            x_right, y_bottom = min(self.W_final, x_left + W_target), min(self.H_final, y_top + H_target)
            final_img_array[y_top:y_bottom, x_left:x_right] = resized_img_array
            return Image.fromarray(final_img_array)


class CIFARLoader(NoAnnotationImageLoader):
    def __init__(self,
                 size: Tuple[int, int],
                 p: float,
                 background_generator: Callable,
                 resize_method: str = 'long'
                 ):

        super().__init__(size=size, p=p, background_generator=background_generator, resize_method=resize_method)

    def __call__(self, img: Image):
        img = img.convert("RGB")

        if self.resize_method == 'adjust':

            h_resized, w_resized = int(self.H_final * self.p), int(self.W_final * self.p)
            resizer = torchvision.transforms.Resize((h_resized, w_resized))
            return resizer(img)

        elif self.resize_method == 'long':

            H, W = img.size[1], img.size[0]
            W_target = self.W_final * self.p
            H_target = W_target * H / W
            H_target, W_target = int(H_target), int(W_target)
            background = self.background_generator((self.H_final, self.W_final))
            resized_img = torchvision.transforms.Resize((H_target, W_target))(img)
            resized_img_array = np.array(resized_img)
            final_img_array = np.array(background)
            x_left, y_top = int((self.W_final - W_target) / 2), int((self.H_final - H_target) / 2)
            x_right, y_bottom = min(self.W_final, x_left + W_target), min(self.H_final, y_top + H_target)
            final_img_array[y_top:y_bottom, x_left:x_right] = resized_img_array
            return Image.fromarray(final_img_array)


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

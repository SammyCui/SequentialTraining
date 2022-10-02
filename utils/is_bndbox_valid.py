import json
import os
import argparse
from typing import Tuple
from PIL import Image
import xmltodict
import sys

import warnings

warnings.filterwarnings('error')
# parser = argparse.ArgumentParser(description="Check if boundbox.xml is valid")
# parser.add_argument('--func', type=str, required=True)
# parser.add_argument('--path', type=str, required=True)
# parser.add_argument('--remove', type=bool, required=False)
#
# args = parser.parse_args()


def get_bndbox_size(annotation_path: str, cat: str) -> Tuple[int, int, int, int]:
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


def is_bndbox_big(anno_path: str, threshold):
    cat = anno_path.split('/')[-2]
    x_min, y_min, x_max, y_max = get_bndbox_size(annotation_path=anno_path, cat=cat)
    h_bb_max, w_bb_max = y_max - y_min, x_max - x_min
    return (h_bb_max > threshold or w_bb_max > threshold) and (h_bb_max > 0) and (w_bb_max > 0)


def validate_annotations(annotation_path: str, image_path: str) -> bool:
    """
    check if the annotation file is correct, e.g. no 0 side,
    :param annotation_path: path to annotation file, has to be of the form: */category/*.xml
    :param image_path: path to image file
    :return:
    """

    category = annotation_path.split('/')[-2]
    x_min, y_min, x_max, y_max = get_bndbox_size(annotation_path=annotation_path,
                                                 cat=category)
    w, h = Image.open(image_path).size
    with open(annotation_path, "r") as xml_obj:
        my_dict = xmltodict.parse(xml_obj.read())
        xml_obj.close()
    # get the image size from the corresponding annotation file, to check if they match or not.
    annotation_img_size = my_dict['annotation']['size']

    return (eval(annotation_img_size['width']) == w) and (eval(annotation_img_size['height']) == h) \
           and (x_min > 0) and (y_min > 0) and (x_max > 0) and (y_max > 0) \
           and (x_max <= w) and (y_max <= h)


def is_bndbox_valid(annotation_path: str, cat: str, verbose: bool = False) -> bool:
    valid = True
    if not os.path.isfile(annotation_path):
        if verbose:
            print(annotation_path, ' does not exist')
        return
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
    if y_max_best - y_min_best <= 0:
        valid = False
        print(f'{annotation_path} y axis of bndbox is invalid.')
    if x_max_best - x_min_best <= 0:
        valid = False
        print(f'{annotation_path} x axis of bndbox is invalid.')

    return valid


def is_image_bndbox_valid(image_root, anno_root, dataset_name: str = 'Imagenet', remove=False):
    """

    :param image_root:
    :param anno_root:
    :param dataset_name:
    :param remove:
    :return:
    """
    for dir in os.listdir(image_root):
        dir_path = os.path.join(image_root, dir)

        if os.path.isdir(dir_path):
            print(f'Scanning {dir}...')
            if dataset_name == 'Imagenet':
                xml_root = os.path.join(anno_root, dir, 'Annotation', dir)
            else:
                xml_root = os.path.join(anno_root, dir)
            count_valid = 0
            count_invalid = 0
            for img in os.listdir(dir_path):
                if img.endswith(('.JPEG', '.jpg')):
                    xml_path = os.path.join(xml_root, f"{img.split('.')[0]}.xml")
                    if not os.path.isfile(xml_path):
                        continue
                    if (not is_bndbox_big(xml_path, 150)) or \
                            (not validate_annotations(os.path.join(xml_root, f"{img.split('.')[0]}.xml"),
                                                 os.path.join(dir_path, img))):
                        count_invalid += 1
                        if remove:
                            os.remove(xml_path)
                    else:
                        count_valid += 1
            print('Valid: ', count_valid, 'Invalid: ', count_invalid)


def is_image_bndbox_correct(image_root, anno_root, dataset_name: str = 'Imagenet', remove=False):
    """
    check if the imagenet annotation is minimumly correct, e.g. map to the correct image and numbers > 0
    :param dataset_name:
    :param image_root:
    :param anno_root:
    :param remove:
    :return:
    """
    for dir in os.listdir(image_root):
        dir_path = os.path.join(image_root, dir)

        if os.path.isdir(dir_path):
            print(f'Scanning {dir}...')
            if dataset_name == 'Imagenet':
                xml_root = os.path.join(anno_root, dir, 'Annotation', dir)
            else:
                xml_root = os.path.join(anno_root, dir)
            count_valid = 0
            count_invalid = 0
            for img in os.listdir(dir_path):
                if img.endswith(('.JPEG', '.jpg')):
                    xml_path = os.path.join(xml_root, f"{img.split('.')[0]}.xml")
                    if not os.path.isfile(xml_path):
                        continue
                    if not validate_annotations(xml_path,
                                           os.path.join(dir_path, img)):
                        count_invalid += 1
                        if remove:
                            os.remove(xml_path)
                    else:
                        count_valid += 1
            print('Valid: ', count_valid, 'Invalid: ', count_invalid)


def remove_bad_images(image_root: str, remove: bool = False):
    for cat in os.listdir(image_root):
        if not os.path.isdir(os.path.join(image_root, cat)):
            continue
        print(f'==>Scanning {cat}')
        for image in os.listdir(os.path.join(image_root, cat)):
            img_path = os.path.join(image_root, cat, image)
            try:
                if image.endswith(('.JPEG', '.jpg', '.png')):
                    img = Image.open(img_path)
                    img.close()
            except UserWarning:
                print('Corrupt EXIF data on file: ', )
                if remove:
                    os.remove(img_path)
            except Exception:
                print('Bad image: ', img_path)
                if remove:
                    os.remove(img_path)


def coco_count_big_bbox(path:str, threshold: int = 150):
    with open(path) as json_file:
        img_dict = json.load(json_file)
    for key, val in img_dict.items():
        big_objs = [x for x in val if (x['bbox'][2] >= int(threshold)) or (x['bbox'][3] >= int(threshold))]
        print(key, ': ', len(val), len(big_objs))


if __name__ == '__main__':
    args = sys.argv
    globals()[args[1]](*args[2:])

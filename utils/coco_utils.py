import json
from typing import List
from tqdm import tqdm
import requests
from pycocotools.coco import COCO
import os
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3 import Retry
import argparse
import pickle


class COCOTools:
    def __init__(self, path_to_json: str):
        self.coco = COCO(path_to_json)

    def coco_downloader(self,
                        img_root: str,
                        classes: List[str] = None,
                        images_per_class: int = None,
                        n_classes: int = 40) -> None:

        available_classes = self.coco.loadCats(self.coco.getCatIds())
        available_classes = [cat['name'] for cat in available_classes]
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        if not classes:
            if n_classes:
                class_to_download = available_classes[:n_classes]
            else:
                class_to_download = available_classes
        else:
            class_to_download = classes
        for cat in class_to_download:
            print('==> Downloading: ', cat)
            catIds = self.coco.getCatIds(catNms=[cat])

            imgIds = self.coco.getImgIds(catIds=catIds)
            if images_per_class:
                np.random.seed(40)
                np.random.shuffle(imgIds)
                imgIds = imgIds[:images_per_class]

            images = self.coco.loadImgs(imgIds)

            for im in tqdm(images):
                if not os.path.isfile(os.path.join(img_root, im['file_name'])):
                    img_data = session.get(im['coco_url']).content
                    with open(os.path.join(img_root, im['file_name']), 'wb') as handler:
                        handler.write(img_data)

    def build_classification_json(self, image_root: str, save_path: str):
        class_dict = {}
        id2class_dict = self.get_id2class_dict(self.coco)
        n_files = len(os.listdir(image_root))
        with tqdm(total=n_files) as pbar:
            for filename in os.listdir(image_root):
                # get annotation ids from the filename, which is also its id
                annIDs = self.coco.getAnnIds(int(filename.split('.')[0].lstrip('0')))
                anns = self.coco.loadAnns(annIDs)
                for ann in anns:
                    category_name = id2class_dict[(ann['category_id'])]
                    if category_name not in class_dict:
                        class_dict[category_name] = []
                    class_dict[category_name].append({'path': os.path.join(image_root, filename),
                                                      'bbox': ann['bbox'],
                                                      'category': category_name,
                                                      'category_id': int(ann['category_id'])})
                pbar.update(1)

        num_objects = 0
        for key, val in class_dict.items():
            num_objects += len(val)
            print(key, len(val))
        print('number of classes: ', len(class_dict.keys()))
        print('number of objects: ', num_objects)

        with open(save_path, 'w') as file:
            json.dump(class_dict, file)
        return class_dict

    @staticmethod
    def get_id2class_dict(coco_obj) -> dict:
        cats = coco_obj.loadCats(coco_obj.getCatIds())
        id2class_dict = {}
        for cat in cats:
            id2class_dict[cat['id']] = cat['name']

        return id2class_dict


def get_big_coco_classes(input_size, path_to_json, min_image_per_class, num_classes=None, save_path=None):
    classes = []
    with open(path_to_json) as json_file:
        img_dict = json.load(json_file)
    for key, val in img_dict.items():
        big_objs = [x for x in val if (x['bbox'][2] >= input_size) or (x['bbox'][3] >= input_size)]
        if len(big_objs) > min_image_per_class:
            classes.append(key)
    classes = sorted(classes)
    if num_classes:
        classes = classes[:num_classes]
    if save_path:
        with open(save_path, "wb") as fp:
            pickle.dump(classes, fp)
    return classes



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main module to run')
    parser.add_argument('--path_to_json', type=str, help="path to coco annotation json file")
    parser.add_argument('--image_root', type=str, const=None, default=None, nargs='?', help="path to the dir of all images")
    parser.add_argument('--save_path', type=str, const=None, default=None, nargs='?', help="path to the file for saving the json obj")
    args = parser.parse_args()
    cocotool = COCOTools(args.path_to_json)
    #cocotool.coco_downloader(img_root='../datasets/coco', classes=['sandwich', 'tv', 'book'], images_per_class=1)
    cocotool.build_classification_json(image_root=args.image_root, save_path=args.save_path)

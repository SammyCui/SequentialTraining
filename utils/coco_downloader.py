import os
from typing import List
import numpy as np
from pycocotools.coco import COCO
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from tqdm.notebook import tqdm
import sys
import argparse

parser = argparse.ArgumentParser(description='COCO downloader')
parser.add_argument('--annotation_path', type=str, help='path to COCO annotation json file')
parser.add_argument('--img_root', type=str, help='path to the directory where you want images to be downloaded to')
parser.add_argument('--classes', type=list, default=None, help='a list of classes that you want to download')
parser.add_argument('--images_per_class', type=int, default=None, help='number of images you want to downlaod per class')
parser.add_argument('--n_classes', type=int, help='number of classes you want to download')
args = parser.parse_args()


def coco_downloader(annotation_path: str,
                    img_root: str,
                    classes: List[str] = None,
                    images_per_class: int = None,
                    n_classes: int = 40) -> None:
    coco = COCO(annotation_path)
    available_classes = coco.loadCats(coco.getCatIds())
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
        catIds = coco.getCatIds(catNms=[cat])

        imgIds = coco.getImgIds(catIds=catIds)
        if images_per_class:
            np.random.seed(40)
            np.random.shuffle(imgIds)
            imgIds = imgIds[:images_per_class]

        images = coco.loadImgs(imgIds)

        for im in tqdm(images):
            if not os.path.isfile(os.path.join(img_root,im['file_name'])):
                img_data = session.get(im['coco_url']).content
                with open(os.path.join(img_root,im['file_name']), 'wb') as handler:
                    handler.write(img_data)


if __name__ == '__main__':
    # coco_downloader('/Users/xuanmingcui/Downloads/instances_val2017.json',
    # '/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTraining/datasets/coco', None, 10, 3)

    coco_downloader(annotation_path=args.annotation_path,
                    img_root=args.img_root,
                    classes=args.classes,
                    images_per_class=args.images_per_class,
                    n_classes=args.n_classes
                    )

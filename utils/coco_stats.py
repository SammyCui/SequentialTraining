from pycocotools.coco import COCO

import os
import argparse

parser = argparse.ArgumentParser(description='COCO list classes')
parser.add_argument('--annotation_path', type=str, help='path to COCO annotation json file')
parser.add_argument('--image_path', type=str, help='path to COCO images dir')

args = parser.parse_args()


coco = COCO(args.annotation_path)

for filename in os.listdir(args.image_path):
    coco.getAnnIds(filename.split('.')[0])

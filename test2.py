import argparse
import os
from PIL import Image
from torch_datasets import ImagenetFolder, VOCImageFolder

import torch, torchvision
from pipeline_imagenet import RunModel
from data_utils import GenerateBackground, IsValidFileImagenet, VOCDistancingImageLoader
from pathlib import Path
import sys

sys.path.append('/usr/local/lib/python3.6/site-packages')
sys.path.append('/usr/lib64/python3.6/site-packages')
sys.path.append('/u/erdos/students/xcui32/.local/lib/python3.6/site-packages')
sys.path.append('/usr/lib64/python3.6/site-packages')

parser = argparse.ArgumentParser(description="Check if boundbox.xml is valid")
parser.add_argument('--cat', type=str, required=True)
parser.add_argument('--n', type=int, required=False)
parser.add_argument('--p', type=float, required=False)
parser.add_argument('--save_path', type=str, required=False)


args = parser.parse_args()

train_root_path = Path(__file__).parent / "datasets/VOC2012/VOC2012_filtered/train/root"
val_root_path = Path(__file__).parent / "datasets/VOC2012_filtered/val"
test_root_path = Path(__file__).parent / "datasets/VOC2012_filtered/test"

root = '/u/erdos/cnslab/imagenet'
anno = '/u/erdos/cnslab/imagenet-bndbox/bndbox'
bg = GenerateBackground(bg_type='color', bg_color=(0, 0, 0))
count = 0
is_valid_file = IsValidFileImagenet(anno, 150)
for file in os.listdir(os.path.join(root, args.cat)):
    if not is_valid_file(os.path.join(root, args.cat, file)):
        continue
    if count >= args.n:
        break
    print(file)
    loader = VOCDistancingImageLoader(size=(150,150),p=args.p, background_generator=bg, annotation_root_path=os.path.join(anno,args.cat,'Annotation'))
    image = loader(os.path.join(root, args.cat, file))
    #image = transform(image)
    if not os.path.exists(f'/u/erdos/students/xcui32/cnslab/results/images/{args.cat}'):
        os.mkdir(f'/u/erdos/students/xcui32/cnslab/results/images/{args.cat}')
    image.save(f'/u/erdos/students/xcui32/cnslab/results/images/{args.cat}/{file}')
    count += 1

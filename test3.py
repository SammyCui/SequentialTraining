import argparse
import os
from PIL import Image
from torch_dataset_06 import ImagenetFolder

import torch, torchvision
from pipeline_imagenet import RunModel
from data_utils import GenerateBackground, IsValidFileImagenet, VOCDistancingImageLoader
from pathlib import Path

train_root_path = "/u/erdos/cnslab/imagenet"
train_anno_root_path = "/u/erdos/cnslab/imagenet-bndbox/bndbox"
result_dirpath = Path(__file__).parent / "results/Imagenet90BlackR182folds"
device = "cuda:0" if torch.cuda.is_available() else "cpu"
# leatherback_turtle, tiger_shark, common_newt, tree_frog, thunder_snake, scorpion, barn_spider, centipede,
# prairie_chicken, jellyfish
cls_to_use = ['n01665541', 'n01491361', 'n01630670', 'n01644373', 'n01728572', 'n01770393', 'n01773549', 'n01784675',
              'n01798484', 'n01910747']
testing_distances_all = [0.2, 0.4, 0.6, 0.8, 1]
print(device)
pipeline_params = {'train_root_path': train_root_path,
                   'train_anno_path': train_anno_root_path,
                   'dataset_name': 'imagenet',
                   'cls_to_use': cls_to_use,
                   'num_classes': None,
                   'target_distances': [0.2, 0.4, 0.6, 0.8, 1],
                   'testing_distances': testing_distances_all,
                   'test_size': 0.2,
                   'image_per_class': 2,
                   'background': GenerateBackground(bg_type='color', bg_color=(0, 0, 0)),
                   'size': (150, 150),
                   'device': device,
                   'llo_targets': [0.2, 0.4, 0.6, 0.8],
                   'n_folds_to_use': 2,
                   'verbose': 0,
                   'resize_method': 'long',
                   'epochs': 200,
                   'n_folds': 5,
                   'batch_size': 128,
                   'num_workers': 16,
                   'model_name': 'resnet18',
                   'result_dirpath': result_dirpath,
                   'random_seed': 40,
                   'save_checkpoints': True,
                   'save_progress_checkpoints': False}

optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9}
print('Initializing indices ...')

pipeline_params['is_valid_file'] = IsValidFileImagenet(train_anno_root_path, threshold=pipeline_params['size'][0])
dataset = ImagenetFolder(cls_to_use=pipeline_params['cls_to_use'],
                         root=train_root_path,
                         num_classes=pipeline_params['num_classes'],
                         image_per_class=pipeline_params['image_per_class'],
                         is_valid_file=pipeline_params['is_valid_file'])
print(dataset.classes)
print('-----')
print(dataset.class_to_idx)
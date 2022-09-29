import os
import sys
from collections import Counter
from typing import List

import numpy as np

from torch_dataset_06 import ImagenetFolder

sys.path.append('/usr/local/lib/python3.6/site-packages')
sys.path.append('/usr/lib64/python3.6/site-packages')
sys.path.append('/u/erdos/students/xcui32/.local/lib/python3.6/site-packages')

import torch, torchvision
from pipeline_imagenet import RunModel
from data_utils import GenerateBackground, IsValidFileImagenet
from pathlib import Path


def main():
    train_root_path = "/u/erdos/cnslab/imagenet-distinct"
    train_anno_root_path = "/u/erdos/cnslab/imagenet-bndbox/bndbox"
    result_dirpath = Path(__file__).parent / "results/Imagenet40R18Black"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # leatherback_turtle, tiger_shark, common_newt, tree_frog, thunder_snake, scorpion, barn_spider, centipede,
    # prairie_chicken, jellyfish
    cls_to_use = ['n01665541', 'n01491361', 'n01630670', 'n01644373', 'n01728572', 'n01770393', 'n01773549',
                  'n01784675', 'n01798484', 'n01910747']
    testing_distances_all = [0.2, 0.4, 0.6, 0.8, 1]
    target_distances = [0.2, 0.4, 0.6, 0.8, 1]
    print(device)
    pipeline_params = {'train_root_path': train_root_path,
                       'train_anno_path': train_anno_root_path,
                       'dataset_name': 'imagenet',
                       'cls_to_use': None,  # cls_to_use,
                       'num_classes': None,
                       'target_distances': target_distances,
                       'testing_distances': testing_distances_all,
                       'test_size': 0.2,
                       'image_per_class': None,
                       'background': GenerateBackground(bg_type='color', bg_color=(0, 0, 0)),
                       'size': (150, 150),
                       'device': device,
                       'llo_targets': None,
                       'n_folds_to_use': None,
                       'verbose': 0,
                       'resize_method': 'long',
                       'epochs': 200,
                       'n_folds': 5,
                       'batch_size': 128,
                       'num_workers': 16,
                       'model_name': 'resnet18',
                       'result_dirpath': result_dirpath,
                       'random_seed': 40,
                       'save_checkpoints': False,
                       'save_progress_checkpoints': False}

    print('Initializing indices ...')

    pipeline_params['is_valid_file'] = IsValidFileImagenet(train_anno_root_path, threshold=pipeline_params['size'][0])
    dataset = ImagenetFolder(cls_to_use=pipeline_params['cls_to_use'],
                             root=train_root_path,
                             num_classes=pipeline_params['num_classes'],
                             image_per_class=pipeline_params['image_per_class'],
                             is_valid_file=pipeline_params['is_valid_file'])

    indices = list(range(len(dataset)))
    np.random.seed(pipeline_params['random_seed'])
    np.random.shuffle(indices)
    train_indices = indices[:int((1 - pipeline_params['test_size']) * len(indices))]
    test_indices = indices[int((1 - pipeline_params['test_size']) * len(indices)):]
    pipeline_params['train_indices'] = train_indices
    pipeline_params['test_indices'] = test_indices
    print(dict(Counter(dataset.targets)))

    print('number of train: ', len(train_indices))
    print('number of test: ', len(test_indices))

if __name__ == '__main__':
    main()
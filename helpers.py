from typing import List, Optional, Tuple, Iterable, Union

import torch
import torchvision
import numpy as np
from .loader import ResizeImageLoader, NoAnnotationImageLoader, CIFARLoader
from .utils.data_utils import GenerateBackground, IsValidFileImagenet
from .datasets_legacy import VOCDataset, ImagenetDataset, CIFAR10Dataset, available_datasets
from torch.utils.data import Dataset, SubsetRandomSampler, ConcatDataset, DataLoader, Subset


def get_dataset(dataset_name, size, p, image_roots: List[str], annotation_roots: Optional[List[str]],
                indices: List[int] = None,
                resize_method: str = 'long',
                cls_to_use: List['str'] = None,
                num_classes: int = None,
                num_samples_to_use: int = None,
                random_seed: int = 40, return_classes: bool = False):
    if dataset_name not in available_datasets:
        raise Exception('Provided dataset name not available. Either check spelling or implement that dataset.')
    background = GenerateBackground(bg_type='color', bg_color=(0, 0, 0))
    dataset_list = []
    ImageDataset = eval(f"{dataset_name}Dataset")
    if 'CIFAR' in dataset_name:
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)  # Cifar dataset mean & std
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # For all others, adopt Imagenet mean & std

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])

    if annotation_roots is None:
        if 'CIFAR' in dataset_name:
            loader = CIFARLoader(size, p, background, resize_method)
        else:
            loader = NoAnnotationImageLoader(size, p, background, resize_method)
        for image_root in image_roots:
            if 'CIFAR' in dataset_name:
                dataset = ImageDataset(root=image_root, loader=loader, transform=transform)
            else:
                dataset = ImageDataset(root=image_root, loader=loader, transform=transform, cls_to_use=cls_to_use,
                                       num_classes=num_classes)

            if indices:
                dataset = Subset(dataset, indices=indices)
            dataset_list.append(dataset)
    else:

        for anno_root, image_root in zip(annotation_roots, image_roots):
            loader = ResizeImageLoader(size, p, anno_root, background, dataset_name)
            if 'Imagenet' in dataset_name:
                is_valid_file = IsValidFileImagenet(anno_root=anno_root, threshold=size[0])
                dataset = ImageDataset(root=image_root, loader=loader, transform=transform, cls_to_use=cls_to_use,
                                       num_classes=num_classes, is_valid_file=is_valid_file)
            else:
                dataset = ImageDataset(root=image_root, loader=loader, transform=transform, cls_to_use=cls_to_use,
                                       num_classes=num_classes)
            if indices:
                dataset = Subset(dataset, indices=indices)
            dataset_list.append(dataset)
    dataset_classes = dataset_list[0].classes
    concat_dataset = ConcatDataset(dataset_list)
    if num_samples_to_use:
        subset_indices = list(range(len(concat_dataset)))
        np.random.seed(random_seed)
        np.random.shuffle(subset_indices)
        subset_indices = subset_indices[:num_samples_to_use]
        if return_classes:
            return Subset(concat_dataset, subset_indices), dataset_classes
        else:
            return Subset(concat_dataset, subset_indices)
    else:
        if return_classes:
            return concat_dataset, dataset_classes
        else:
            return concat_dataset


def get_cv_indices(num_samples: int, n_folds: int = 5, n_folds_to_use: int = 5, random_seed: int = 40):
    val_size = 1 / n_folds
    cv_indices = list(range(num_samples))
    np.random.seed(random_seed)
    np.random.shuffle(cv_indices)
    split = int(np.floor(val_size * num_samples))

    for fold in range(n_folds_to_use):
        split1 = int(np.floor(fold * split))
        split2 = int(np.floor((fold + 1) * split))
        val_idx = cv_indices[split1:split2]
        train_idx = np.append(cv_indices[:split1], cv_indices[split2:])
        train_idx = train_idx.astype('int32')
        yield train_idx, val_idx


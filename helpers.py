from typing import List, Optional, Tuple, Iterable, Union

import torch
import torchvision
import numpy as np
from .loader import ResizeImageLoader, NoAnnotationImageLoader, CIFARLoader
from .utils.data_utils import GenerateBackground, IsValidFileImagenet
from .datasets_legacy import VOCDataset, ImagenetDataset, CIFAR10Dataset, available_datasets, COCODataset
from torch.utils.data import Dataset, SubsetRandomSampler, ConcatDataset, DataLoader, Subset
import torchvision.transforms.functional as F


class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, 0, 'constant')


def get_dataset(dataset_name, size: Tuple[int, int], p, image_roots: List[str], annotation_roots: Optional[List[str]],
                indices: List[int] = None,
                resize_method: str = 'long',
                cls_to_use: List['str'] = None,
                num_classes: int = None,
                num_samples_to_use: int = None,
                min_image_per_class: int = None,
                max_image_per_class: int = None,
                train: bool = True,
                origin: bool = False,
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

    if (size[0] - int(size[0] * p)) % 2 == 0:
        padding_0, padding_1 = int((size[0] - int(size[0] * p)) / 2), int((size[0] - int(size[0] * p)) / 2)
    else:
        padding_0, padding_1 = int((size[0] - int(size[0] * p)) / 2), int((size[0] - int(size[0] * p)) / 2) + 1
    if origin:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((int(size[0] * p), int(size[0] * p))),
            torchvision.transforms.Pad((padding_0, padding_0, padding_1, padding_1)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean, std),
        ])

    if annotation_roots is None:
        if origin:
            loader = None
        else:
            if 'CIFAR' in dataset_name:
                loader = CIFARLoader(size, p, background, resize_method)
            else:
                loader = NoAnnotationImageLoader(size, p, background, resize_method)
        for image_root in image_roots:
            if 'CIFAR' in dataset_name:
                dataset = ImageDataset(root=image_root, loader=loader, transform=transform, train=train, download=True)
            else:

                dataset = ImageDataset(root=image_root, loader=loader, transform=transform, cls_to_use=cls_to_use,
                                       num_classes=num_classes)
            dataset_classes = dataset.classes
            if indices:
                dataset = Subset(dataset, indices=indices)
            dataset_list.append(dataset)

    else:

        for anno_root, image_root in zip(annotation_roots, image_roots):
            loader = None if origin else ResizeImageLoader(size, p, anno_root, background, dataset_name)
            if 'Imagenet' in dataset_name:
                is_valid_file = IsValidFileImagenet(anno_root=anno_root, threshold=size[0])
                dataset = ImageDataset(root=image_root, loader=loader, transform=transform, cls_to_use=cls_to_use,
                                       num_classes=num_classes, is_valid_file=is_valid_file)
            elif 'COCO' in dataset_name:
                if origin:
                    raise Exception('origin does not support coco dataset')
                dataset = COCODataset(root=image_root, path_to_json=anno_root, num_classes=num_classes,
                                      cls_to_use=cls_to_use,
                                      input_size=size, size=p, resize_method=resize_method, transform=transform,
                                      min_image_per_class=min_image_per_class, max_image_per_class=max_image_per_class)

            else:
                dataset = ImageDataset(root=image_root, loader=loader, transform=transform, cls_to_use=cls_to_use,
                                       num_classes=num_classes)
            dataset_classes = dataset.classes
            if indices:
                dataset = Subset(dataset, indices=indices)
            dataset_list.append(dataset)

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

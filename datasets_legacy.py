import json
import os
from typing import Tuple, Optional, Callable, Any, List, Dict, Iterable, cast

from PIL import Image
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
from torchvision.datasets.cifar import CIFAR10
import torchvision
import re
from torch.utils.data import Dataset

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


"""
This is the dataset module corresponds to torchvision==0.6.0, as on erdos cluster
"""


available_datasets = ['VOC', 'Imagenet', 'CIFAR10']


class VOCDataset(torchvision.datasets.VisionDataset):

    def __init__(self,
                 root: str,
                 cls_to_use: Optional[Iterable[str]] = None,
                 num_classes: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 ):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.cls_to_use = cls_to_use
        self.loader = loader
        self.is_valid_file = is_valid_file
        self.num_classes = num_classes
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS if is_valid_file is None else None, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Modified version of torchvision.datasets.folder.find_classes
        to select a subset of image classes
        """
        if self.cls_to_use is not None:
            classes = sorted(
                entry.name for entry in os.scandir(directory) if entry.is_dir() and entry.name in self.cls_to_use)
        else:
            classes = sorted(
                entry.name for entry in os.scandir(directory) if entry.is_dir())
            if self.num_classes:
                classes = classes[:self.num_classes]
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self._find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            count = 0
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
                        count += 1

                        if target_class not in available_classes:
                            available_classes.add(target_class)

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances


class ImagenetDataset(torchvision.datasets.VisionDataset):

    def __init__(self, cls_to_use: Optional[Iterable[str]],
                 root: str,
                 num_classes: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 image_per_class: Optional[int] = None,
                 ):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.cls_to_use = cls_to_use
        self.loader = loader
        self.num_classes = num_classes
        self.image_per_class = image_per_class
        self.is_valid_file = is_valid_file
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS if is_valid_file is None else None, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                IMG_EXTENSIONS)))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def _find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Modified version of torchvision.datasets.folder.find_classes
        to select a subset of image classes
        """
        if self.cls_to_use is not None:
            classes = sorted(
                entry.name for entry in os.scandir(directory) if entry.is_dir() and entry.name in self.cls_to_use)
        else:

            classes = []
            # for idx, entry in enumerate(os.scandir(directory)):
            #     if self.num_classes:
            #         if idx >= self.num_classes:
            #             break
            #     if entry.is_dir() and re.fullmatch(r"n\d{8}", entry.name):
            #         classes.append(entry.name)
            for entry in os.scandir(directory):
                if entry.is_dir() and re.fullmatch(r"n\d{8}", entry.name):
                    classes.append(entry.name)

            classes = sorted(classes)
            if self.num_classes:
                classes = classes[:self.num_classes]

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def make_dataset(
        self,
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        directory = os.path.expanduser(directory)

        if class_to_idx is None:
            _, class_to_idx = self._find_classes(directory)
        elif not class_to_idx:
            raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

        both_none = extensions is None and is_valid_file is None
        both_something = extensions is not None and is_valid_file is not None
        if both_none or both_something:
            raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

        if extensions is not None:
            def is_valid_file(x: str) -> bool:
                return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

        is_valid_file = cast(Callable[[str], bool], is_valid_file)

        instances = []
        available_classes = set()
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(directory, target_class)
            if not os.path.isdir(target_dir):
                continue
            count = 0
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)
                        count += 1
                        if self.image_per_class and (count >= self.image_per_class):
                            break

                        if target_class not in available_classes:
                            available_classes.add(target_class)
                if self.image_per_class and (count >= self.image_per_class):
                    break

        empty_classes = set(class_to_idx.keys()) - available_classes
        if empty_classes:
            msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
            if extensions is not None:
                msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
            raise FileNotFoundError(msg)

        return instances

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class CIFAR10Dataset(CIFAR10):
    def __init__(self,
                 root,
                 loader=None,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False):

        super().__init__(root, transform=transform, train=train, target_transform=target_transform, download=download)
        self.loader = loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        if self.loader:
            img = self.loader(img)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class COCODataset(Dataset):
    def __init__(self,
                 root: str,
                 path_to_json: str,
                 transform,
                 num_classes,
                 size_threshold,
                 min_image_per_class,
                 max_image_per_class):

        self.root = root
        self.transform = transform
        with open(path_to_json) as json_file:
            img_dict = json.load(json_file)

        self.classes = []
        for key, val in img_dict.items():
            big_objs = [x for x in val if (x['bbox'][2] >= size_threshold) or (x['bbox'][3] >= size_threshold)]
            if len(big_objs) > min_image_per_class:
                self.classes.append(key)
        self.classes = sorted(self.classes)
        self.img_dict = img_dict
        if num_classes:
            self.classes = self.classes[:num_classes]
        self.img_dict = {}
        self.data = []
        for cls in self.classes:
            if max_image_per_class:
                self.img_dict[cls] = img_dict[cls][:max_image_per_class]
                self.data.extend(img_dict[cls][:max_image_per_class])
            else:
                self.img_dict[cls] = img_dict[cls]
                self.data.extend(img_dict[cls])

        self.class_idx = list(range(len(self.classes)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        meta_info = self.data[idx]
        img = Image.open(meta_info['path'])
        if self.transform is not None:
            img = self.transform(img)

        target = meta_info['category_id']

        return img, target











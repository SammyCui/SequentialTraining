import json
import os
from typing import Tuple, Optional, Callable, Any, List, Dict, Iterable, cast

from PIL import Image
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
from torchvision.datasets.cifar import CIFAR10
import torchvision
import re
from torch.utils.data import Dataset
from utils.data_utils import GenerateBackground
from utils.coco_utils import get_big_coco_classes
import numpy as np

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")

"""
This is the dataset module corresponds to torchvision==0.6.0, as on erdos cluster
"""

available_datasets = ['VOC', 'Imagenet', 'CIFAR10', 'COCO']


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
        samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS if is_valid_file is None else None,
                                    is_valid_file)
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
        samples = self.make_dataset(self.root, class_to_idx, IMG_EXTENSIONS if is_valid_file is None else None,
                                    is_valid_file)
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
                 input_size: Tuple,
                 size,
                 resize_method,
                 cls_to_use,
                 min_image_per_class,
                 max_image_per_class):

        """

        :param root:
        :param path_to_json:
        :param transform:
        :param num_classes:
        :param input_size:
        :param size:
        :param resize_method: adjust, long, original
        :param min_image_per_class:
        :param max_image_per_class:
        """

        self.root = root
        self.transform = transform
        self.input_size = input_size
        self.H_final, self.W_final = input_size
        self.resize_method = resize_method
        self.size = size
        with open(path_to_json) as json_file:
            img_dict = json.load(json_file)

        self.classes = []
        if cls_to_use:
            self.classes = cls_to_use
        else:
            self.classes = get_big_coco_classes(input_size=self.H_final, path_to_json=path_to_json, min_image_per_class=min_image_per_class,
                                                num_classes=num_classes)
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
        # bbox
        x_min, y_min, x_max, y_max = meta_info['bbox'][0], meta_info['bbox'][1], \
                                     meta_info['bbox'][0] + meta_info['bbox'][2], meta_info['bbox'][1] + meta_info['bbox'][3]
        target = meta_info['category_id']
        if self.resize_method == 'adjust':
            bnd_box_img = img.crop((x_min, y_min, x_max, y_max))
            h_resized, w_resized = int(self.input_size * self.size), int(self.input_size * self.size)
            resizer = torchvision.transforms.Resize((h_resized, w_resized))
            img = resizer(bnd_box_img)
            if self.transform is not None:
                img = self.transform(img)
            return img, target

        elif self.resize_method == 'long':

            H, W = img.size[1], img.size[0]
            background_callable = GenerateBackground(bg_type='color', bg_color=(0, 0, 0))
            background = background_callable(self.input_size, img)
            print(meta_info)
            H_bb, W_bb = meta_info['bbox'][3], meta_info['bbox'][2]
            # longer_side = 'H_bb' if H_bb > W_bb else 'W_bb'

            if H_bb > W_bb:
                # get the size of resized bndbox
                H_bb_target = self.H_final * self.size
                assert H_bb_target <= H_bb, 'This bndbox longest side is smaller than the target side. Cannot resize the target bigger'
                W_bb_target = H_bb_target * W_bb / H_bb
            else:
                # get the size of resized bndbox
                W_bb_target = self.W_final * self.size
                assert W_bb_target <= W_bb, 'This bndbox longest side is smaller than the target side. Cannot resize the target bigger'
                H_bb_target = W_bb_target * H_bb / W_bb

            # get the size of orig image after resizing bndbox
            H_target = int(np.floor(H_bb_target * H / H_bb))
            W_target = int(np.floor(W_bb_target * W / W_bb))
            # resize original image
            resized_img = torchvision.transforms.Resize((H_target, W_target))(img)
            resized_img_array = np.array(resized_img)
            # get coordinates of resized bndbox in the resized image
            x_min_resized, x_max_resized = int(np.floor(x_min / W * W_target)), int(np.floor(x_max / W * W_target))
            y_min_resized, y_max_resized = int(np.floor(y_min / H * H_target)), int(np.floor(y_max / H * H_target))

            # get coordinates in the resized image to crop (bounding box plus the surroundings)
            # from the resized orig image (to place in the final image)

            # if it goes out of the left bound of the resized image, let it be 0
            x_min_orig = max(0, int(np.floor(x_min_resized - (self.W_final - W_bb_target) / 2)))
            # if it goes out of the right bound, let it be the right bound
            x_max_orig = min(W_target, int(np.floor(x_max_resized + (self.W_final - W_bb_target) / 2)))

            # if it goes out of the top bound of the resized image, let it be 0
            y_min_orig = max(0, int(np.floor(y_min_resized - (self.H_final - H_bb_target) / 2)))
            # if it goes out of the bottom bound, let it be the bottom bound
            y_max_orig = min(H_target, int(np.floor(y_max_resized + (self.H_final - H_bb_target) / 2)))
            if y_max_orig - y_min_orig - self.H_final > 1:
                raise Exception(f"Sth. went wrong during resizing for path: {meta_info['path']}" )
            elif 0 < y_max_orig - y_min_orig - self.H_final <= 1:
                y_max_orig -= y_max_orig - y_min_orig - self.H_final
            if x_max_orig - x_min_orig - self.W_final > 1:
                raise Exception(f"Sth. went wrong during resizing for path: {meta_info['path']}")
            elif 0 < x_max_orig - x_min_orig - self.W_final <= 1:
                x_max_orig -= x_max_orig - x_min_orig - self.W_final
            # get coordinates to place the above cropped resized image in the final image

            # 1. get coordinates of resized bndbox in the final image
            x_min_final_bb, x_max_final_bb = int(np.floor((self.W_final - W_bb_target) / 2)), int(
                np.round(self.W_final / 2 + W_bb_target / 2))
            y_min_final_bb, y_max_final_bb = int(np.floor((self.H_final - H_bb_target) / 2)), int(
                np.round(self.H_final / 2 + H_bb_target / 2))

            # 2. Add/subtract surrounding distances of resized-bndbox-in-resized-orig-image to
            # each coordinates of resized bndbox in the final image

            x_min_final, y_min_final = max(0, x_min_final_bb - (x_min_resized - x_min_orig)), max(0, y_min_final_bb - (
                    y_min_resized - y_min_orig))

            x_max_final, y_max_final = x_min_final + x_max_orig - x_min_orig, y_min_final + y_max_orig - y_min_orig
            if x_max_final - 1 == self.W_final:
                x_min_final, x_max_final = x_min_final - 1, x_max_final - 1
            if y_max_final - 1 == self.W_final:
                y_min_final, y_max_final = y_min_final - 1, y_max_final - 1

            # place the resized-orig-image onto the final image, which has only background right now

            final_img_array = np.array(background)

            try:
                final_img_array[int(np.floor(y_min_final)):int(np.floor(y_max_final)),
                int(np.floor(x_min_final)):int(np.floor(x_max_final))] = \
                    resized_img_array[int(np.floor(y_min_orig)):int(np.floor(y_max_orig)),
                    int(np.floor(x_min_orig)):int(np.floor(x_max_orig))]

            except:
                raise ValueError(f"size doesn't match. path: {meta_info['path']}")

            img = Image.fromarray(final_img_array)

            if self.transform is not None:
                img = self.transform(img)
            return img, target


if __name__ == '__main__':
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)  # For all others, adopt Imagenet mean & std

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    dataset = COCODataset(root='./datasets/coco', path_to_json='./datasets/classification_val2017.json',transform=None, cls_to_use=None,
                          num_classes=3, input_size=(150,150), size=1, resize_method='long', min_image_per_class=1, max_image_per_class=5)
    s, target = dataset[0]
    s.show()
import os
from typing import Tuple, Optional, Callable, Any, List, Dict, Iterable, cast
from torchvision.datasets.folder import default_loader, has_file_allowed_extension
import torchvision
import re

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class VOCImageFolder(torchvision.datasets.DatasetFolder):

    def __init__(self, cls_to_use: Optional[Iterable[str]],
                 root: str,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 ):
        self.cls_to_use = cls_to_use
        super().__init__(root, transform=transform,
                         target_transform=target_transform,
                         loader=loader,
                         is_valid_file=is_valid_file,
                         extensions=IMG_EXTENSIONS if is_valid_file is None else None)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Modified version of torchvision.datasets.folder.find_classes
        to select a subset of image classes
        """
        print('here!!!!')
        if self.cls_to_use is not None:
            classes = sorted(
                entry.name for entry in os.scandir(directory) if entry.is_dir() and entry.name in self.cls_to_use)
        else:
            classes = sorted(
                entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


class ImagenetFolder(torchvision.datasets.DatasetFolder):

    def __init__(self, cls_to_use: Optional[Iterable[str]],
                 root: str,
                 num_classes: Optional[int] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 loader: Callable[[str], Any] = default_loader,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 image_per_class: Optional[int] = None,
                 ):
        self.cls_to_use = cls_to_use
        self.num_classes = num_classes
        self.image_per_class = image_per_class
        super().__init__(root, transform=transform,
                         target_transform=target_transform,
                         loader=loader,
                         is_valid_file=is_valid_file,
                         extensions=IMG_EXTENSIONS if is_valid_file is None else None)

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Modified version of torchvision.datasets.folder.find_classes
        to select a subset of image classes
        """
        print('here!!!!!!!!!!!')
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
            if self.num_classes:
                classes = classes[:self.num_classes]
            classes = sorted(classes)

        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        print('inside ',len(classes))
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
            _, class_to_idx = self.find_classes(directory)
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
        print('make')
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

import os
from typing import Tuple, Optional, Callable, Any, List, Dict, Iterable
from torchvision.datasets.folder import default_loader
import torchvision

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

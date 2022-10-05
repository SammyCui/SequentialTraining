import os
import re

from PIL import Image
import torchvision
import numpy as np
from typing import Tuple, Callable, Optional
from utils.data_utils import GenerateBackground
from utils.data_utils import get_bndbox


class ResizeImageLoader:
    """
        simgle data loader for torch datasets. Full doc: https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
        only work for VOC file structure
    """

    def __init__(self, size: Tuple[int, int], p: float, annotation_root_path: str,
                 background_generator: Optional[Callable] = None,
                 dataset_name: str = 'VOC',
                 resize_method: str = 'long',
                 draw_bndbox_width: int = None):
        """

        :param size: size of the full final training image (bounding box + rest-of-image + background)
        :param p: H_bb / size[0] or W_bb / size[1], depending on what resize_method is used
        :param background: string specify which
        :param annotation_root_path:
        :param resize_method: if 'long', resize bounding box to p=max(H_bb,W_bb)/corresponding H/W of size
                              if 'short',                       p=min(H_bb,W_bb)/corresponding H/W of size with center crop
        """
        self.H_final, self.W_final = size
        self.p = p
        self.background_generator = background_generator
        self.annotation_root_path = annotation_root_path
        self.resize_method = resize_method
        self.dataset_name = dataset_name
        self.draw_bndbox_width = draw_bndbox_width

    def __call__(self, path: str):
        img = Image.open(path)
        img = img.convert("RGB")

        path_component = re.split(r'\.|/', path)
        cat, name = path_component[-3], path_component[-2]
        if self.dataset_name == 'Imagenet':
            annotation_path = os.path.join(self.annotation_root_path, cat, 'Annotation', cat, name + '.xml')
        else:
            annotation_path = os.path.join(self.annotation_root_path, cat, name + '.xml')

        x_min, y_min, x_max, y_max = get_bndbox(annotation_path=annotation_path, cat=cat)
        if self.resize_method == 'adjust':
            bnd_box_img = img.crop((x_min, y_min, x_max, y_max))
            h_resized, w_resized = int(self.H_final * self.p), int(self.W_final * self.p)
            resizer = torchvision.transforms.Resize((h_resized, w_resized))
            return resizer(bnd_box_img)

        elif self.resize_method == 'long':

            H, W = img.size[1], img.size[0]
            background = self.background_generator((self.H_final, self.W_final), img)

            H_bb, W_bb = y_max - y_min, x_max - x_min
            # longer_side = 'H_bb' if H_bb > W_bb else 'W_bb'

            if H_bb > W_bb:
                # get the size of resized bndbox
                H_bb_target = self.H_final * self.p
                assert H_bb_target <= H_bb, 'This bndbox longest side is smaller than the target side. Cannot resize the target bigger'
                W_bb_target = H_bb_target * W_bb / H_bb
            else:
                # get the size of resized bndbox
                W_bb_target = self.W_final * self.p
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
                raise Exception("Sth. went wrong during resizing for path: ", path)
            elif 0 < y_max_orig - y_min_orig - self.H_final <= 1:
                y_max_orig -= y_max_orig - y_min_orig - self.H_final
            if x_max_orig - x_min_orig - self.W_final > 1:
                raise Exception("Sth. went wrong during resizing for path: ", path)
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

                if self.draw_bndbox_width:
                    final_img_array[y_min_final_bb:y_min_final_bb + self.draw_bndbox_width,
                    x_min_final_bb:x_max_final_bb] = np.full(
                        (self.draw_bndbox_width, x_max_final_bb - x_min_final_bb, 3), 255, dtype='uint8')
                    final_img_array[y_max_final_bb - self.draw_bndbox_width:y_max_final_bb,
                    x_min_final_bb:x_max_final_bb] = np.full(
                        (self.draw_bndbox_width, x_max_final_bb - x_min_final_bb, 3), 255, dtype='uint8')
                    final_img_array[y_min_final_bb:y_max_final_bb,
                    x_min_final_bb:x_min_final_bb + self.draw_bndbox_width] = np.full(
                        (y_max_final_bb - y_min_final_bb, self.draw_bndbox_width, 3), 255, dtype='uint8')
                    final_img_array[y_min_final_bb:y_max_final_bb,
                    x_max_final_bb - self.draw_bndbox_width:x_max_final_bb] = np.full(
                        (y_max_final_bb - y_min_final_bb, self.draw_bndbox_width, 3), 255, dtype='uint8')

            except:
                raise ValueError(f"size doesn't match. path: {path}")

            return Image.fromarray(final_img_array)

    def crop(self):
        pass


class NoAnnotationImageLoader:
    def __init__(self,
                 size: Tuple[int, int],
                 p: float,
                 background_generator: Callable,
                 resize_method: str = 'long'
                 ):

        self.H_final, self.W_final = size
        self.p = p
        self.background_generator = background_generator
        self.resize_method = resize_method

    def __call__(self, path: str):
        img = Image.open(path)
        img = img.convert("RGB")

        if self.resize_method == 'adjust':

            h_resized, w_resized = int(self.H_final * self.p), int(self.W_final * self.p)
            resizer = torchvision.transforms.Resize((h_resized, w_resized))
            return resizer(img)

        elif self.resize_method == 'long':

            H, W = img.size[1], img.size[0]
            if H > W:
                H_target = self.H_final * self.p
                W_target = H_target * W / H
            else:
                W_target = self.W_final * self.p
                H_target = W_target * H / W
            H_target, W_target = int(H_target), int(W_target)
            background = self.background_generator((self.H_final, self.W_final))
            resized_img = torchvision.transforms.Resize((H_target, W_target))(img)
            resized_img_array = np.array(resized_img)
            final_img_array = np.array(background)
            x_left, y_top = int((self.W_final - W_target) / 2), int((self.H_final - H_target) / 2)
            x_right, y_bottom = min(self.W_final, x_left + W_target), min(self.H_final, y_top + H_target)
            final_img_array[y_top:y_bottom, x_left:x_right] = resized_img_array
            return Image.fromarray(final_img_array)


class CIFARLoader(NoAnnotationImageLoader):
    def __init__(self,
                 size: Tuple[int, int],
                 p: float,
                 background_generator: Callable,
                 resize_method: str = 'long'
                 ):

        super().__init__(size=size, p=p, background_generator=background_generator, resize_method=resize_method)

    def __call__(self, img: Image):
        img = img.convert("RGB")

        if self.resize_method == 'adjust':

            h_resized, w_resized = int(self.H_final * self.p), int(self.W_final * self.p)
            resizer = torchvision.transforms.Resize((h_resized, w_resized))
            return resizer(img)

        elif self.resize_method == 'long':

            H, W = img.size[1], img.size[0]
            W_target = self.W_final * self.p
            H_target = W_target * H / W
            H_target, W_target = int(H_target), int(W_target)
            background = self.background_generator((self.H_final, self.W_final))
            resized_img = torchvision.transforms.Resize((H_target, W_target))(img)
            resized_img_array = np.array(resized_img)
            final_img_array = np.array(background)
            x_left, y_top = int((self.W_final - W_target) / 2), int((self.H_final - H_target) / 2)
            x_right, y_bottom = min(self.W_final, x_left + W_target), min(self.H_final, y_top + H_target)
            final_img_array[y_top:y_bottom, x_left:x_right] = resized_img_array
            return Image.fromarray(final_img_array)


if __name__ == '__main__':
    background_callable = GenerateBackground(bg_type='color', bg_color=(0, 0, 0))
    loader = ResizeImageLoader(size=(150,150), p=0.5, annotation_root_path='/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTraining/datasets/VOC2012_filtered/train/annotations',
                               background_generator=background_callable)
    img = loader('/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTraining/datasets/VOC2012_filtered/train/root/aeroplane/2008_000037.jpg')
    img.show()
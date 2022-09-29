import os
from PIL import Image


def sanity_check(root: str, anno: str):
    """
    check if image is valid. Delete both img and anno if not
    :param root: image
    :param anno: anotations
    :return:
    """

    for cat in os.listdir(root):
        for img in os.listdir(os.path.join(root, cat)):
            name = img.split(".")[0]
            try:
                Image.open(os.path.join(root, cat, img))
            except:
                print(f"{os.path.join(root, cat, img)} is not available.")
                os.remove(os.path.join(root, cat, img))
                os.remove(os.path.join(anno, cat, name + '.xml'))


if __name__ == '__main__':
    sanity_check('/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/train/root',
                 '/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/train/annotations')
    sanity_check('/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/val/root',
                 '/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/val/annotations')
    sanity_check('/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/test/root',
                 '/u/erdos/students/xcui32/cnslab/datasets/VOC2012/VOC2012_filtered/test/annotations')

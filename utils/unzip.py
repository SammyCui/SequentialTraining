import zipfile
import sys
import tarfile
import os
import re


def unzip(filename, target):
    if 'tar.gz' in filename:
        file = tarfile.open(filename)
    elif 'zip' in filename:
        file = zipfile.ZipFile(filename, 'r')
    else:
        raise Exception('File type not supported')

    file.extractall(target)
    file.close()


def unzip_images(path:str):
    """
    extract .tar files in the dir
    :param dir:
    :return:
    """

    filenames = os.listdir(path)
    for filename in filenames:
        if 'tar.gz' in filename or '.tar' in filename:
            file_id_list = [x for x in filenames if filename.split('.')[0] == x]
            if len(file_id_list) == 0:
                file = tarfile.open(os.path.join(path, filename))
            else:
                continue
        elif 'zip' in filename:
            file = zipfile.ZipFile(os.path.join(path, filename), 'r')
        elif len(filename.split()) == 1:
            continue
        else:
            raise Exception('File type not supported')
        print('Extracting: ', filename)
        file.extractall(os.path.join(path,filename.split('.')[0]))
        file.close()


def unzip_classes(bndbox_dir, img_dir, delete: bool = True):
    """
    unzip bounding boxes
    :param bndbox_dir:
    :param img_dir:
    :param delete:
    :return:
    """
    img_classes = [entry.name for entry in os.scandir(img_dir) if entry.is_dir() and re.fullmatch(r"n\d{8}", entry.name)]
    bndbox_classes = [entry.name.split('.')[0] for entry in os.scandir(bndbox_dir) if re.fullmatch(r"n\d{8}.tar.gz", entry.name)]
    for img_class in img_classes:
        if img_class in bndbox_classes:
            if os.path.exists(os.path.join(bndbox_dir, img_class)): continue
            file = tarfile.open(os.path.join(bndbox_dir, img_class + '.tar.gz'))
            print(f'Unzipping {file}')
            file.extractall(os.path.join(bndbox_dir, img_class))
            file.close()
            if delete:
                os.remove(os.path.join(bndbox_dir, img_class + '.tar.gz'))
        else:
            print(f'No bndbox found for: {img_class}')

if __name__ == '__main__':
    args = sys.argv
    globals()[args[1]](*args[2:])
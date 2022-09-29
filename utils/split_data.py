import os
import re
import pandas as pd
import numpy as np
import pickle as pkl
import shutil

# index images into train/val/test with different labels
cats = ['aeroplane', 'bicycle', 'bird', 'boat', 'car', 'cat', 'train', 'tvmonitor']

cats_label = dict(zip(cats, [i for i in range(1, len(cats) + 1)]))
layout_dir = '//datasets/VOC2012/ImageSets/Main'
image_dir = '//datasets/VOC2012/JPEGImages'
annotation_dir = '//datasets/VOC2012/Annotations'
annotation_dir_target = '//datasets/VOC/annotations'
image_dir_target = '//datasets/VOC/root'

for filename in os.listdir(layout_dir):
    split = re.split(r'\.|_', filename)
    cat, which_set = split[0], split[1]
    if len(split) == 3 and (which_set == 'trainval'):
        file = pd.read_csv(layout_dir + '/' + filename, header=None, sep=r"\s+")
        true_samples = file[file[1] == 1][0].to_numpy()
        if not os.path.isdir(f'{image_dir_target}/{cat}'):
            os.mkdir(f'{image_dir_target}/{cat}')
        if not os.path.isdir(f'{annotation_dir_target}/{cat}'):
            os.mkdir(f'{annotation_dir_target}/{cat}')
        for sample_name in true_samples:
            if os.path.exists(f"{image_dir}/{sample_name}.jpg") and os.path.exists(f"{annotation_dir}/{sample_name}.xml"):
                shutil.copyfile(f"{image_dir}/{sample_name}.jpg", f"{image_dir_target}/{cat}/{sample_name}.jpg")
                shutil.copyfile(f"{annotation_dir}/{sample_name}.xml", f"{annotation_dir_target}/{cat}/{sample_name}.xml")
            else:
                print("File not exist: ", sample_name)

#
#
# print(result.shape)

# pkl.dump((cats_label, result), open('voc_labels.p', 'wb'))

# # check if all the indexing is valid
#


# # print(datasets['train'].shape, datasets['val'].shape, datasets['test'].shape)
# anno_set = set([filename.split('.')[0] for filename in os.listdir('/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTrainingCNN/datasets/VOC2012/Annotations')])
# image_set = set([filename.split('.')[0] for filename in os.listdir('/Users/xuanmingcui/Documents/projects/cnslab/cnslab/SequentialTrainingCNN/datasets/VOC2012/JPEGImages')])
#
# for filename,label in result:
#     if filename not in anno_set or filename not in image_set:
#         print(filename)


# pkl.dump(result, open( "voc_labels.p", "wb" ) )
#
# print(datasets['train'][:,0])
#
#

#
# print(datasets['train'].shape, datasets['val'].shape, datasets['test'].shape)
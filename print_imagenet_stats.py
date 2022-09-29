from collections import Counter

from data_utils import VOCDistancingImageLoader, GenerateBackground
from torch_datasets import ImagenetFolder

test_loader = VOCDistancingImageLoader((150,150), p=0.2,
                                        background_generator=GenerateBackground(bg_type = 'color'),
                                        annotation_root_path='/u/erdos/cnslab/imagenet-bndbox/bndbox/')
test_dataset = ImagenetFolder(cls_to_use=None, root='/u/erdos/cnslab/imagenet/',
                              loader=test_loader)
print('n of classes: ', len(test_dataset.classes))
train_dist = {k:dict(Counter(test_dataset.targets))[v] for k,v in test_dataset.class_to_idx.items()}
print(test_dataset.classes)
print('avg number of image per class: ', sum(train_dist.values())/len(test_dataset.classes))

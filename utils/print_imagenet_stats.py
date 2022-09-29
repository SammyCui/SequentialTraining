from collections import Counter

from utils.data_utils import GenerateBackground
from loader import ResizeImageLoader
from datasets_legacy import ImagenetDataset

test_loader = ResizeImageLoader((150, 150), p=0.2,
                                background_generator=GenerateBackground(bg_type = 'color'),
                                annotation_root_path='/u/erdos/cnslab/imagenet-bndbox/bndbox/')
test_dataset = ImagenetDataset(cls_to_use=None, root='/u/erdos/cnslab/imagenet/',
                              loader=test_loader)
print('n of classes: ', len(test_dataset.classes))
train_dist = {k:dict(Counter(test_dataset.targets))[v] for k,v in test_dataset.class_to_idx.items()}
print(test_dataset.classes)

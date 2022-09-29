from pycocotools.coco import COCO
import argparse

parser = argparse.ArgumentParser(description='COCO list classes')
parser.add_argument('--annotation_path', type=str, help='path to COCO annotation json file')
parser.add_argument('--first_n', type=int, help='first n classes')

args = parser.parse_args()


coco = COCO(args.annotation_path)
available_classes = coco.loadCats(coco.getCatIds())
available_classes = [cat['name'] for cat in available_classes]

print('==>all: ')
print(available_classes)
print('\n')


print(f'==>first {args.first_n}: ')
print(available_classes[:args.first_n])
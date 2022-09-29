import argparse

from utils.miscs import none_or_str

parser = argparse.ArgumentParser(description='Main module to run')

parser.add_argument('--test', type=none_or_str, nargs='?', default='2', const=None, help='path to result directory')
parser.add_argument('--sizes', default=150, const=150, type=int, nargs='?',help='width/height of input image size. Default 150 -- (150, 150)')

args = parser.parse_args()

# args.sizes = args.sizes.split(',')
for arg, val in vars(args).items():
    print(arg, val, type(val))
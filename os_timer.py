import os
import argparse
import time

parser = argparse.ArgumentParser(description="Time performance of os")
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--num_dirs', type=int, required=False)

args = parser.parse_args()


def time_os(root, num_dirs):
    start = time.time()
    count = 0
    for dir in os.listdir(root):
        count += 1
        if count >= num_dirs:
            break
    end = time.time()
    print(end-start)


if __name__ == '__main__':
    time_os(args.path, args.num_dirs)

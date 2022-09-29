import os

import urllib.request

url = "https://gist.githubusercontent.com/aaronpolhamus/964a4411c0906315deb9f4a3723aac57/raw/aa66dd9dbf6b56649fa3fab83659b2acbf3cbfd1/map_clsloc.txt"
file = urllib.request.urlopen(url)

class_dict = {}

for line in file:
    line = line.decode("utf-8")
    index, id, name = line.split(' ')
    class_dict[index] = name

n = 0
for folder in os.scandir('/u/erdos/cnslab/imagenet'):

    if os.path.isdir(folder.path):
        try:
            print(folder.name, '----', class_dict[folder.name])
        except:
            print(f'Not found: {folder.name}')

        n += 1
        if n == 19:
            print('20 classes above')

print('Number of Classes: ', n)
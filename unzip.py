import zipfile

import sys

args = sys.argv

zip_ref = zipfile.ZipFile(args[1], 'r') #Opens the zip file in read mode
zip_ref.extractall(args[2])
zip_ref.close()
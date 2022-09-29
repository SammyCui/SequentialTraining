import numpy
import torch,torchvision

import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(device)

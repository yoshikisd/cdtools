from __future__ import division, print_function, absolute_import

import CDTools
from CDTools.tools import cmath
import h5py
import torch as t
import numpy as np

with h5py.File('../example_data/NiCr.cxi','r') as f:
    dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(f)

model = CDTools.models.SimplePtycho.from_dataset(dataset)

# Uncomment these to use on the CPU
# default is CPU with 32-bit floats
model.to(device='cuda')
dataset.to(device='cuda')
dataset.get_as(device='cuda')

for loss in model.Adam_optimize(500, dataset):
    print(loss)

from matplotlib import pyplot as plt

probe = cmath.torch_to_complex(model.probe.detach().cpu())
obj = cmath.torch_to_complex(model.obj.detach().cpu())

plt.imshow(np.abs(probe))
plt.colorbar()
plt.figure()
plt.imshow(np.abs(obj))
plt.colorbar()
plt.show()


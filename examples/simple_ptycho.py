from __future__ import division, print_function, absolute_import

import CDTools
from CDTools.tools import cmath
from CDTools.tools.plotting import *
import h5py
import torch as t
import numpy as np

filename = '../../../Projects/CSX_3_19/cxis/processed/114429_p.cxi'
filename = '../../../Projects/CSX_3_19/cxis/processed/115145_p.cxi'
filename = '../../../Downloads/AuBalls_700ms_30nmStep_3_3SS_filter.cxi'


with h5py.File(filename,'r') as f:
    dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(f)


model = CDTools.models.SimplePtycho.from_dataset(dataset)


# Uncomment these to use on the CPU
# default is CPU with 32-bit floats
model.to(device='cuda')
#dataset.to(device='cuda')
dataset.get_as(device='cuda')

#model.probe.requires_grad = False
for loss in model.Adam_optimize(10, dataset):
    print(loss)

#for loss in model.Adam_optimize(20, dataset, lr=0.0005):
#    print(loss)

from matplotlib import pyplot as plt

plot_amplitude(model.probe)
plot_phase(model.probe)
plot_amplitude(model.obj)
plot_phase(model.obj)
plt.show()

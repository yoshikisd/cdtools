from __future__ import division, print_function, absolute_import

import CDTools
from CDTools.tools import cmath
from CDTools import tools
from CDTools.tools.plotting import *
from matplotlib import pyplot as plt
import pickle
from time import time
import datetime

import h5py
import torch as t
import numpy as np


# This file is too large to be distributed via Github.
# Please contact Abe Levitan (alevitan@mit) if you would like access
filename = '/media/Data Bank/CSX_10_18/Processed_CXIs/110531_p.cxi'

with h5py.File(filename,'r') as f:
    dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(f)


# A hack for the specular geometry
dataset.detector_geometry['basis'] = np.array([[0,-30e-6],[30e-6,0],[0,0]])

# Figure out why the padding doesn't work here
model = CDTools.models.FancyPtycho.from_dataset(dataset,randomize_ang = np.pi/4, padding=0, translation_scale=10)#, n_modes=2, propagation_distance=-5e-4)


# Uncomment these to use on the CPU
# default is CPU with 32-bit floats
model.to(device='cuda')
dataset.get_as(device='cuda')


for i, loss in enumerate(model.Adam_optimize(250, dataset,batch_size=5)):
    print(i,loss)
    model.inspect(dataset)


model.inspect(dataset)
model.compare(dataset)
plt.show()

from __future__ import division, print_function, absolute_import

import CDTools
from CDTools.tools import cmath
from CDTools.tools.plotting import *
import h5py
import torch as t
import numpy as np
from matplotlib import pyplot as plt
import pickle


#filename = '../../Downloads/114429_p.cxi'
#filename = '../../../Projects/CSX_3_19/cxis/processed/114429_p.cxi'
filename = '../../../Projects/CSX_3_19/cxis/processed/115145_p.cxi'
#filename = '../../Desktop/Reconstructions/114429_p.cxi'


with h5py.File(filename,'r') as f:
    dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(f)


model = CDTools.models.SimplePtycho.from_dataset(dataset)

# Uncomment these to use on the CPU
# default is CPU with 32-bit floats
model.to(device='cpu')
#dataset.to(device='cuda')
dataset.get_as(device='cpu')

for i, loss in enumerate(model.ePIE(10, dataset)):
    print(i, loss)

for i, loss in enumerate(model.Adam_optimize(10, dataset)):
    print(i, loss)

model.inspect()

#with open('test_results.pickle', 'wb') as f:
#    pickle.dump(model.save_results(),f)

plt.show()

from __future__ import division, print_function, absolute_import

import CDTools
from CDTools.tools.plotting import *
from CDTools.tools.cmath import *
from CDTools.tools import interactions
import h5py
import numpy as np
from matplotlib import pyplot as plt
import torch as t
import pickle

filename = '../../../Downloads/AuBalls_700ms_30nmStep_3_3SS_filter.cxi'
#filename = '/media/Data Bank/CSX_3_19/Processed_CXIs/115195_p.cxi'

with h5py.File(filename,'r') as f:
    dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(f)
    #darks = np.array(f['entry_1/instrument_1/detector_1/data_dark'])

#old_patterns = dataset.patterns.clone()
#dataset.patterns -= t.tensor(np.nanmean(darks,axis=0))
#dataset.patterns = t.clamp(dataset.patterns,min=0)

model = CDTools.models.FancyPtycho.from_dataset(dataset,n_modes=3,randomize_ang=0.1*np.pi)
#dataset.patterns = old_patterns

# default is CPU with 32-bit floats
model.to(device='cuda')
dataset.get_as(device='cuda')

#model.translation_offsets.requires_grad = False

for i, loss in enumerate(model.Adam_optimize(30, dataset, batch_size=100)):
    model.inspect(dataset)
    print(i,loss)

for i, loss in enumerate(model.Adam_optimize(30, dataset, batch_size=100, lr=0.001)):
    model.inspect(dataset)
    print(i,loss)

for i, loss in enumerate(model.Adam_optimize(30, dataset, batch_size=100, lr=0.0001)):
    model.inspect(dataset)
    print(i,loss)


#with open('test_results.pickle', 'wb') as f:
#    pickle.dump(model.save_results(dataset),f)
    
model.inspect(dataset)
plt.show()
exit()

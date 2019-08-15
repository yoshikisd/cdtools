from __future__ import division, print_function, absolute_import

import CDTools
import h5py
import numpy as np
import pickle
from matplotlib import pyplot as plt

filename = 'example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'

with h5py.File(filename,'r') as f:
    dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(f)


model = CDTools.models.FancyPtycho.from_dataset(dataset, n_modes=2)



# default is CPU with 32-bit floats
model.to(device='cuda')
dataset.get_as(device='cuda')


for i, loss in enumerate(model.Adam_optimize(30, dataset, batch_size=100)):
    print(i,loss)
    # Here we see how to liveplot the results - this call will create
    # or update a readout of the various parameters being reconstructed
    model.inspect(dataset)


with open('example_reconstructions/gold_balls.pickle', 'wb') as f:
    pickle.dump(model.save_results(dataset),f)
    
model.inspect(dataset)
dataset.inspect()
model.compare(dataset)
plt.show()

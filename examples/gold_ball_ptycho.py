from __future__ import division, print_function, absolute_import

import CDTools
from CDTools.tools.plotting import *
from CDTools.tools.cmath import *
from CDTools.tools import interactions
import h5py
import numpy as np
from matplotlib import pyplot as plt

filename = '../../../Downloads/AuBalls_700ms_30nmStep_3_3SS_filter.cxi'
#filename = '/media/Data Bank/CSX_3_19/Processed_CXIs/115195_p.cxi'

with h5py.File(filename,'r') as f:
    dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(f)
    darks = np.array(f['entry_1/instrument_1/detector_1/data_dark'])

old_patterns = dataset.patterns.clone()
dataset.patterns -= t.tensor(np.nanmean(darks,axis=0))
dataset.patterns = t.clamp(dataset.patterns,min=0)

model = CDTools.models.FancyPtycho.from_dataset(dataset,n_modes=3,randomize_ang=0.1*np.pi)
dataset.patterns = old_patterns

# default is CPU with 32-bit floats
model.to(device='cuda')
dataset.get_as(device='cuda')

#model.translation_offsets.requires_grad = False

for i, loss in enumerate(model.Adam_optimize(30, dataset, batch_size=100)):
    print(i,loss)

for i, loss in enumerate(model.Adam_optimize(30, dataset, batch_size=100, lr=0.001)):
    print(i,loss)

for i, loss in enumerate(model.Adam_optimize(50, dataset, batch_size=100, lr=0.0001)):
    print(i,loss)


# Show some figures of merit
plot_amplitude(model.probe[0], basis=model.probe_basis.cpu()*1e6)
plot_phase(model.probe[0], basis=model.probe_basis.cpu()*1e6)
plot_amplitude(model.obj, basis=model.probe_basis.cpu()*1e6)
plot_phase(model.obj, basis=model.probe_basis.cpu()*1e6)
translations = (interactions.translations_to_pixel(model.probe_basis.cpu(), dataset.translations) + model.translation_offsets.detach().cpu()).numpy()
plt.figure()
plt.plot(translations[:,1],translations[:,0],'k-',linewidth=0.5)
plt.plot(translations[:,1],translations[:,0],'b.')
plt.show()

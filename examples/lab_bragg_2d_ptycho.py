from __future__ import division, print_function, absolute_import

import CDTools
from matplotlib import pyplot as plt
import pickle


#
# This is the testing file for the Bragg ptychography code that includes
# corrections for probe propagation and for off-axis far field diffraction
# I'm starting to build it up using the lab ptychography forward data so I
# can initially test that it works on "softball" data before moving to the
# actual Bragg geometry data
#

#filename = 'example_data/lab_ptycho_data.cxi'
filename = '/media/Data Bank/Lab Ptycho/Zone Plate Bragg 633.cxi'
dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(filename)

#dataset.inspect()

model = CDTools.models.Bragg2DPtycho.from_dataset(dataset,probe_support_radius=60)#propagate_probe=False)#, n_modes=8)
model.to(device='cuda')
dataset.get_as(device='cuda')


model.translation_offsets.requires_grad = False
for i, loss in enumerate(model.Adam_optimize(100, dataset)):
    model.inspect(dataset)
    print(i,loss)
    
#with open('example_reconstructions/lab_bragg_2d_ptycho.pickle', 'wb') as f:
#    pickle.dump(model.save_results(dataset),f)

#model.compare(dataset)
plt.show()

from __future__ import division, print_function, absolute_import

import CDTools
from matplotlib import pyplot as plt
import pickle

filename = 'example_data/lab_ptycho_data.cxi'
dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(filename)

# dataset.inspect()
# plt.show()

model = CDTools.models.FancyPtycho.from_dataset(dataset, oversampling=2,
                                                probe_support_radius=90)
model.to(device='cuda')
dataset.get_as(device='cuda')


model.translation_offsets.requires_grad = False
for i, loss in enumerate(model.Adam_optimize(50, dataset)):
    model.inspect(dataset)
    print(i,loss)
    
with open('example_reconstructions/lab_ptycho.pickle', 'wb') as f:
    pickle.dump(model.save_results(dataset),f)

model.compare(dataset)
plt.show()

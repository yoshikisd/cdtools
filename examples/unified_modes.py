from __future__ import division, print_function, absolute_import

import CDTools
from matplotlib import pyplot as plt
import pickle

filename = 'example_data/lab_ptycho_data.cxi'
dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(filename)

# dataset.inspect()
# plt.show()

# dm_rank=-1 tells it to use a full-rank unified mode approximation
model = CDTools.models.FancyPtycho.from_dataset(dataset, oversampling=1,n_modes=3, dm_rank=-1)


model.to(device='cuda')
dataset.get_as(device='cuda')

model.translation_offsets.requires_grad = False
for i, loss in enumerate(model.Adam_optimize(20, dataset)):
    model.inspect(dataset)
    print(i,loss)

model.tidy_probes()

for i, loss in enumerate(model.Adam_optimize(20, dataset, lr=0.0001)):
    model.inspect(dataset)
    print(i,loss)

model.tidy_probes(normalize=True)
model.inspect(dataset)
                         
with open('example_reconstructions/unified_modes.pickle', 'wb') as f:
    pickle.dump(model.save_results(dataset),f)

model.compare(dataset)
plt.show()

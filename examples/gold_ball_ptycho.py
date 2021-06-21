from __future__ import division, print_function, absolute_import

import CDTools
from matplotlib import pyplot as plt
import pickle
import time

# First, we load an example dataset from a .cxi file
filename = 'example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'
dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(filename)

# Next, we create a ptychography model from the dataset
# Note that we explicitly as for two incoherent probe modes
model = CDTools.models.FancyPtycho.from_dataset(dataset, n_modes=2,dm_rank=0, probe_support_radius=50)

# Let's do this reconstruction on the GPU, shall we? 
model.to(device='cuda')
dataset.get_as(device='cuda')

for loss in model.Adam_optimize(100, dataset, batch_size=50, schedule=True):
    # And we liveplot the updates to the model as they happen
    print(model.report())
    model.inspect(dataset)

# And we save the reconstruction out to a file
#with open('example_reconstructions/gold_balls.pickle', 'wb') as f:
#    pickle.dump(model.save_results(dataset),f)

model.tidy_probes()

# Finally, we plot the results
model.inspect(dataset)
#model.compare(dataset)
plt.show()

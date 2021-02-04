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
model = CDTools.models.FancyPtycho.from_dataset(dataset, n_modes=2)

# Let's do this reconstruction on the GPU, shall we? 
model.to(device='cuda')
dataset.get_as(device='cuda')

for i, loss in enumerate(model.Adam_optimize(20, dataset, batch_size=50)):
#for i, loss in enumerate(model.LBFGS_optimize(20, dataset, lr=1, history_size=5)):
    # And we liveplot the updates to the model as they happen
    print(i,loss)
    model.inspect(dataset)

# And we save the reconstruction out to a file
#with open('example_reconstructions/gold_balls.pickle', 'wb') as f:
#    pickle.dump(model.save_results(dataset),f)

# Finally, we plot the results
#model.inspect(dataset)
#model.compare(dataset)
#plt.show()

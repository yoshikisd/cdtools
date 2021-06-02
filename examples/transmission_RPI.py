from __future__ import division, print_function, absolute_import

import CDTools
from matplotlib import pyplot as plt
import pickle
from torch.utils.data import Subset

# First, we load an example dataset from a .cxi file

ss_filename = 'example_data/Optical_Data_ss.cxi'

#with open('example_data/Optical_ptycho.pickle', 'rb') as f:
with open('example_data/Optical_ptycho_incoherent.pickle', 'rb') as f:
    ptycho_results = pickle.load(f)

probe = ptycho_results['probe']
background = ptycho_results['background']

dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(ss_filename)


# Next, we create a ptychography model from the dataset
# Note that we explicitly as for two incoherent probe modes
model = CDTools.models.RPI.from_dataset(dataset, probe, [900,900],
                                        background=background, n_modes=2,
                                        initialization='random')


# Let's do this reconstruction on the GPU, shall we? 
model.to(device='cuda')
dataset.get_as(device='cuda')

# Note that the inspect step takes the vast majority of the time
# The regularization is an L2 regularizer that empirically helps accelerate
# convergence
for i, loss in enumerate(model.LBFGS_optimize(30, dataset, lr=0.4, regularization_factor=[0.05,0.05])):#0.1)):
    model.inspect(dataset)
    print(i,loss)
    
#model.inspect(dataset)
    
# Now we use the regularizer to damp all but the top modes
for i, loss in enumerate(model.LBFGS_optimize(50, dataset, lr=0.4, regularization_factor=[0.001,0.1])):
    #model.inspect(dataset)
    print(i,loss)

results = model.save_results()


# Finally, we plot the results
model.inspect(dataset)
model.compare(dataset)
plt.show()

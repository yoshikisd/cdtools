from __future__ import division, print_function, absolute_import
import CDTools
from matplotlib import pyplot as plt
# First, we load an example dataset from a .cxi file
filename = '/media/Data Bank/APS_HXN_07_19/CXIs/scan_94513_cxi.h5'

#filename = 'data/scan_94361_cxi.h5'
dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(filename)
# Next, we create a ptychography model from the dataset
model = CDTools.models.Simple3DBPP.from_dataset(dataset)
#import pdb; pdb.set_trace()
model.inspect(dataset)
plt.show()

# # Now, we run a short reconstruction from the dataset!
# for i, loss in enumerate(model.Adam_optimize(1, dataset)):
#     print(i, loss)

from __future__ import division, print_function, absolute_import

import CDTools
from matplotlib import pyplot as plt

# First, we load an example dataset from a .cxi file
filename = 'example_data/lab_ptycho_data.cxi'
dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(filename)

# Next, we create a ptychography model from the dataset
model = CDTools.models.SimplePtycho.from_dataset(dataset)

# Now, we run a short reconstruction from the dataset!
for i, loss in enumerate(model.Adam_optimize(10, dataset)):
    print(i, loss)

# Finally, we plot the results
model.inspect(dataset)
model.compare(dataset)
plt.show()

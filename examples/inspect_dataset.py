from __future__ import division, print_function, absolute_import

import CDTools
from matplotlib import pyplot as plt

# First, we load an example dataset from a .cxi file
filename = 'example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'
dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(filename)

# And we take a look at the data
dataset.inspect()
plt.show()

import cdtools
from matplotlib import pyplot as plt

# First, we load an example dataset from a .cxi file
filename = 'example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

# And we take a look at the data
dataset.inspect()
plt.show()

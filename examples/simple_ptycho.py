from __future__ import division, print_function, absolute_import

import CDTools
from CDTools.tools import cmath
from CDTools.tools.plotting import *
from matplotlib import pyplot as plt
import pickle

filename = 'example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'

dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(filename)

model = CDTools.models.SimplePtycho.from_dataset(dataset)

for i, loss in enumerate(model.Adam_optimize(10, dataset)):
    print(i, loss)

for i, loss in enumerate(model.Adam_optimize(10, dataset, lr=0.001)):
    print(i, loss)

model.inspect(update=False)
model.compare(dataset)

#with open('test_results.pickle', 'wb') as f:
#    pickle.dump(model.save_results(),f)

plt.show()

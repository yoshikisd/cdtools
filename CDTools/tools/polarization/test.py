from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from copy import copy
import h5py
import pathlib
from CDTools.datasets import CDataset, Ptycho2DDataset, PolarizedPtycho2DDataset
from CDTools.tools import data as cdtdata, initializers, polarization, interactions
from CDTools.tools import plotting
from torch.utils import data as torchdata
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import ticker
from CDTools.models import FancyPtycho, PolarizedFancyPtycho
from torch.utils import data as torchdata

dataset = PolarizedPtycho2DDataset.from_cxi('test_ptycho.cxi')
model = PolarizedFancyPtycho.from_dataset(dataset)

# print('probe', model.probe.shape)
# print('object', model.obj.shape)
# print('transl', dataset.translations.shape)
# print('polarizer', dataset.polarizer.shape)
# print('analyzer', dataset.analyzer.shape)

exit_waves = model.interaction(*dataset[[0]][0])
print(exit_waves.shape)
# sim_patterns = model.forward(*dataset[:][0])

for loss in model.Adam_optimize(10, dataset, batch_size=1, lr=0.002, schedule=True):
    # And we liveplot the updates to the model as they happen
    print(model.report())


# probe = t.rand(5, 2, 10, 12)
# obj = t.rand(2, 2, 10, 12)
# polarizer = t.rand(7)
# analyzer = t.rand(7)

# pol_probes = polarization.apply_linear_polarizer(probe, polarizer)

# exit_waves = interactions.ptycho_2D_sinc(
#     probe, obj, translations,
#     shift_probe=True, multiple_modes=True, polarized=True)

# analyzed_exit_waves = polarization.apply_linear_polarizer(exit_waves, analyzer)

# print(analyzed_exit_waves.shape)
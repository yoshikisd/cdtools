import cdtools
from matplotlib import pyplot as plt
import torch as t
from torch.utils import data as torchdata
import time

# First, we load an example dataset from a .cxi file
filename = 'example_data/lab_ptycho_data.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

# Next, we create a ptychography model from the dataset
model = cdtools.models.FancyPtycho.from_dataset(
    dataset,
    n_modes=3, # Use 3 incoherently mixing probe modes
    oversampling=2, # Simulate the probe on a 2xlarger real-space array
    probe_support_radius=120, # Force the probe to 0 outside a radius of 120 pix
    propagation_distance=5e-3, # Propagate the initial probe guess by 5 mm
    translation_scale=2, # Increase the strength of the position refinement
    units='mm', # Set the units for the live plots
)

device = 'cuda'
model.to(device=device)
dataset.get_as(device=device)

# The learning rate parameter sets the alpha for Adam.
# The beta parameters are (0.9, 0.999) by default
# The batch size sets the minibatch size
for loss in model.Adam_optimize(50, dataset,  lr=0.02,batch_size=10):
    print(model.report())
    model.inspect(dataset)

# It's totally okay to chain several different reconstructions. Here, we
# started with an aggressive refinement to find the probe, and now we
# polish the reconstruction with a lower learning rate and larger minibatch
for loss in model.Adam_optimize(50, dataset,  lr=0.005, batch_size=50):
    print(model.report())
    model.inspect(dataset)

model.inspect(dataset)
model.compare(dataset)
plt.show()

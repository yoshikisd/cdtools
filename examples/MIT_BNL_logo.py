import cdtools
from matplotlib import pyplot as plt


# This file is too large to be distributed via Github.
# Please contact Abe Levitan (alevitan@mit) if you would like access
filename = '79511_p.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

# In this dataset, the edges of the patterns are masked off anyway
# We can easily just remove this data instead of leaving it to float.
dataset.patterns = dataset.patterns[:,70:-70,70:-70]
dataset.mask = dataset.mask[70:-70,70:-70]

# This model definition includes lots of tweaks, described below.
#
# randomize_ang defines the initial random phase noise's extent
# translations_scale defines how aggressive the position reconstruction is
# n_modes is the number of incoherent modes
# propagation_distance is the distance to propagate from the SHARP-style guess of the probe's focal spot (in this case, the value comes from knowledge of the experimental geometry).
model = cdtools.models.FancyPtycho.from_dataset(dataset,
                                                translation_scale = 4,
                                                n_modes=2,
                                                propagation_distance=73e-6)

# Move to the GPU
model.to(device='cuda')
dataset.get_as(device='cuda')


# We turn off position reconstruction for the first phase
model.translation_offsets.requires_grad = False
for loss in model.Adam_optimize(10, dataset, batch_size=15):
    model.inspect(dataset)
    print(model.report())

# And we turn it on for the second phase
model.translation_offsets.requires_grad = True
for loss in model.Adam_optimize(20, dataset, batch_size=15):
    model.inspect(dataset)
    print(model.report())
    
# The third phase lowers the rate further
for loss in model.Adam_optimize(10, dataset, batch_size=15, lr=0.0005):
    model.inspect(dataset)
    print(model.report())

    
model.inspect(dataset)
model.compare(dataset)
plt.show()

from __future__ import division, print_function, absolute_import

import CDTools
from matplotlib import pyplot as plt


# This file is too large to be distributed via Github.
# Please contact Abe Levitan (alevitan@mit) if you would like access
filename = '/media/Data Bank/CSX_6_17/Processed_CXIs/79511_p.cxi'
dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(filename)

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
model = CDTools.models.FancyPtycho.from_dataset(dataset,
                                                translation_scale = 4,
                                                n_modes=2,
                                                propagation_distance=-73e-6)

# Move to the GPU
model.to(device='cuda')
dataset.get_as(device='cuda')


# We turn off position reconstruction for the first phase
model.translation_offsets.requires_grad = False
for i, loss in enumerate(model.Adam_optimize(10, dataset, batch_size=15)):
    model.inspect(dataset)
    print(i,loss)

# And we turn it on for the second phase
model.translation_offsets.requires_grad = True
for i, loss in enumerate(model.Adam_optimize(20, dataset, batch_size=15)):
    model.inspect(dataset)
    print(i,loss)
    
# The third phase lowers the rate further
for i, loss in enumerate(model.Adam_optimize(10, dataset, batch_size=15, lr=0.0005)):
    model.inspect(dataset)
    print(i,loss)

    
model.inspect(dataset)
model.compare(dataset)
plt.show()

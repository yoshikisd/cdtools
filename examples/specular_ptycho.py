from __future__ import division, print_function, absolute_import

import CDTools
from matplotlib import pyplot as plt
import numpy as np


# This file is too large to be distributed via Github.
# Please contact Abe Levitan (alevitan@mit) if you would like access
filename = '/media/Data Bank/CSX_10_18/Processed_CXIs/110531_p.cxi'
dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(filename)


# This model definition includes lots of tweaks, described below.
#
# randomize_ang defines the initial random phase noise's extent
# translations_scale defines how aggressive the position reconstruction is
# scattering_mode overrides any sample normal information stored in the .cxi file
model = CDTools.models.FancyPtycho.from_dataset(dataset,
                                                randomize_ang = np.pi/4,
                                                translation_scale=10,
                                                scattering_mode='reflection')


# Move to the GPU
model.to(device='cuda')
dataset.get_as(device='cuda')


# Run the reconstruction
for i, loss in enumerate(model.Adam_optimize(100, dataset,batch_size=5)):
    print(i,loss)
    model.inspect(dataset)


model.inspect(dataset)
model.compare(dataset)
plt.show()

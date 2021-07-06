import CDTools
from matplotlib import pyplot as plt
from scipy import io

#
# This is the testing dataset for the Bragg ptychography code, that was
# collected with our optical setup using a low NA zone plate illuminating
# a reflective diffraction grating with writing on it.
#

# This file is too large to be distributed via Github.
# Please contact Abe Levitan (alevitan@mit) if you would like access
filename = '/media/Data Bank/Lab Ptycho/Zone Plate Bragg 633.cxi'
dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(filename)

#dataset.inspect()

model = CDTools.models.Bragg2DPtycho.from_dataset(dataset,probe_support_radius=60,correct_tilt=False)
model.to(device='cuda')
dataset.get_as(device='cuda')


model.translation_offsets.requires_grad = False
for loss in model.Adam_optimize(100, dataset):
    model.inspect(dataset)
    print(model.report())

io.savemat('example_reconstructions/lab_bragg_2d_ptycho.mat',
           model.save_results(dataset))

model.compare(dataset)
plt.show()

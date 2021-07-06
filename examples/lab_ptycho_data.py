import CDTools
from matplotlib import pyplot as plt
from scipy import io


# A simple dataset collected from our optical setup
filename = 'example_data/lab_ptycho_data.cxi'
dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(filename)

dataset.inspect(units='mm')
plt.show()

model = CDTools.models.FancyPtycho.from_dataset(dataset, oversampling=2,
                                                probe_support_radius=90,
                                                n_modes=2,
                                                dm_rank=0,units='mm')
model.to(device='cuda')
dataset.get_as(device='cuda')


model.translation_offsets.requires_grad = False
for loss in model.Adam_optimize(50, dataset):
    model.inspect(dataset)
    print(model.report())

model.tidy_probes()

io.savemat('example_reconstructions/lab_ptycho.mat',
           model.save_results(dataset))

model.compare(dataset)
plt.show()

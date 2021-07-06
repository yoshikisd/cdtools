import CDTools
from matplotlib import pyplot as plt
from scipy import io


filename = 'example_data/lab_ptycho_data.cxi'
dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(filename)

#dataset.inspect(units='mm')
#plt.show()

# dm_rank=-1 tells it to use a full-rank unified mode approximation
model = CDTools.models.FancyPtycho.from_dataset(dataset, oversampling=1,n_modes=3, dm_rank=-1)


model.to(device='cuda')
dataset.get_as(device='cuda')

model.translation_offsets.requires_grad = False
for loss in model.Adam_optimize(200, dataset):
    model.inspect(dataset)
    print(model.report())

model.tidy_probes()

for loss in model.Adam_optimize(200, dataset, lr=0.0001):
    model.inspect(dataset)
    print(model.report())

model.tidy_probes(normalize=True)
model.inspect(dataset)
                         
io.savemat('example_reconstructions/unified_modes.mat',
           model.save_results(dataset))

model.compare(dataset)
plt.show()

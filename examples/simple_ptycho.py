import cdtools
from matplotlib import pyplot as plt

# First, we load an example dataset from a .cxi file
filename = 'example_data/lab_ptycho_data.cxi'
dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

# Next, we create a ptychography model from the dataset
model = cdtools.models.SimplePtycho.from_dataset(dataset)

model.to(device='cuda:0')
dataset.get_as(device='cuda:0')

# Now, we run a short reconstruction from the dataset!
for loss in model.Adam_optimize(20, dataset):
    print(model.report())

print(model.save_results().keys())
# Finally, we plot the results
model.inspect(dataset)
model.compare(dataset)
plt.show()

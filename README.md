# CDTools

CDTools is a python library for ptychography and CDI reconstructions, using an Automatic Differentiation based approach.

```python
# imports
from matplotlib import pyplot as plt
from cdtools.datasets import Ptycho2DDataset
from cdtools.models import FancyPtycho

# Load the file
dataset = Ptycho2DDataset.from_cxi('ptycho_data.cxi')

# Generate a model from the data
model = FancyPtycho.from_dataset(dataset)

# Run a reconstruction
for loss in model.Adam_optimize(10, dataset):
    print(model.report())

# And look at the results!
model.inspect(dataset) # See the reconstructed object, probe, etc.
model.compare(dataset) # See how the simulated and measured patterns compare
plt.show()
```

Full installation instructions and documentation can be found [here](https://github.mit.edu/pages/Scattering/cdtools/).

Have a wonderful day!

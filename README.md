# CDTools

CDTools is a python library for ptychography and CDI reconstructions, using an Automatic Differentiation based approach.

```python
# imports
from matplotlib import pyplot as plt
from CDTools.datasets import Ptycho_2D_Dataset
from CDTools.models import SimplePtycho

# Load the file
dataset = Ptycho_2D_Dataset.from_cxi('ptycho_data.cxi')

# Generate a model from the data
model = SimplePtycho.from_dataset(dataset)

# Run a reconstruction
for i, loss in enumerate(model.Adam_optimize(10, dataset)):
    print(i, loss)

# And look at the results!
model.inspect(dataset)
model.compare(dataset)
plt.show()
```

Full installation instructions and documentation can be found [here](https://github.mit.edu/pages/Scattering/CDTools/)

Have a wonderful day!
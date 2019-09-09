.. toctree::
   :hidden:
   :maxdepth: 1

   self
   installation
   examples
   tutorial
   general
   datasets
   models
   tools/index
   indices_tables

Introduction to CDTools
=======================

CDTools is a python library for ptychography and CDI reconstructions, using an Automatic Differentiation based approach.

.. code-block:: python

   # imports
   from matplotlib import pyplot as plt
   from CDTools.datasets import Ptycho2DDataset
   from CDTools.models import SimplePtycho
		
   # Load the file
   dataset = Ptycho2DDataset.from_cxi('ptycho_data.cxi')

   # Generate a model from the data
   model = SimplePtycho.from_dataset(dataset)

   # Run a reconstruction
   for i, loss in enumerate(model.Adam_optimize(10, dataset)):
       print(i, loss)

   # And look at the results!
   model.inspect(dataset)
   model.compare(dataset)
   plt.show()


CDTools makes it simple to load and inspect data stored in .cxi files using python scripts. Several reconstruction models for common geometries are included "out of the box". For more advanced users, it includes a bunch of modular functions for AD ptychography, which can then be used right away from the same scripting framework.

The high-level interface to CDTools is built on a lower level "three-legged stool". This consists of tools to access stored data, tools to visualize data and reconstructions, and tools that implement basic operations relevant to coherent diffraction. All of these tools can be used directly alongside the high-level interface, when needed.

Enough blabber. If you're interested, read the docs!

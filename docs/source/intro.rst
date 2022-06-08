Introduction to CDTools
=======================


.. only:: latex

   Introduction to CDTools
   -----------------------

CDTools is a python library for ptychography and CDI reconstructions, using an Automatic Differentiation (AD) based approach.

.. code-block:: python

   # imports
   import cdtools
   from matplotlib import pyplot as plt
		
   # Load the file
   dataset = cdtools.datasets.Ptycho2DDataset.from_cxi('ptycho_data.cxi')

   # Generate a model from the data
   model = cdtools.models.SimplePtycho.from_dataset(dataset)

   # Run a reconstruction
   for loss in model.Adam_optimize(20, dataset):
       print(model.report())

   # And look at the results!
   model.inspect(dataset)
   model.compare(dataset)
   plt.show()


CDTools makes it simple to load and inspect data stored in .cxi files using python scripts. Several reconstruction models for common geometries are included "out of the box". For more advanced users, it includes a modular functions for AD ptychography. These can can be used to construct new forward models.

The high-level interface to CDTools is built on a lower level "three-legged stool". This consists of tools to access stored data, tools to visualize data and reconstructions, and tools that implement basic operations relevant to coherent diffraction. All of these tools can be used directly alongside the high-level interface, when needed.

Enough blabber. If you're interested, read the docs!


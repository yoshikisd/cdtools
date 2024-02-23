Introduction to CDTools
=======================


.. only:: latex

   Introduction to CDTools
   -----------------------

CDTools is a python library for ptychography and CDI reconstructions, using an automatic differentiation based approach.

.. code-block:: python

   from matplotlib import pyplot as plt
   from cdtools.datasets import Ptycho2DDataset
   from cdtools.models import FancyPtycho
   
   # Load a data file
   dataset = Ptycho2DDataset.from_cxi('ptycho_data.cxi')
   
   # Initialize a model from the data
   model = FancyPtycho.from_dataset(dataset)
   
   # Run a reconstruction
   for loss in model.Adam_optimize(10, dataset):
        print(model.report())
   
   # Save the results
   model.save_to_h5('ptycho_results.h5', dataset)
   
   # And look at them!
   model.inspect(dataset) # See the reconstructed object, probe, etc.
   model.compare(dataset) # See how the simulated and measured patterns compare
   plt.show()


CDTools makes it simple to load and inspect data stored in .cxi files. Reconstruction models for common geometries are included "out of the box". For more advanced users, it includes a collection of differentiable functions which can beused to construct new forward models.

The high-level interface to CDTools - datasets and models - is built on a set oflower-level tools. These include functions for accessing stored data in .cxi files, tools to visualize data and reconstructions, and tools that implement basic operations - such as light propagation - relevant to coherent diffraction. These functions can be used alongside the high-level interface, when needed.

Enough blabber. If you're interested, read the docs!


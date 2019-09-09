Tutorial
========

This tutorial builds on the examples, leading to a more complete understanding of how to use CDTools and - importantly - how to extend it. First, we will cover the details of writing a useful reconstruction script for a particular experiment. Next, we will discuss how to implement a new dataset type for different kinds of coherent diffraction. Finally, we will go over how to make new models to cover specific types of ptychography which aren't described by any of the built-in models.


Reconstruction Scripts
----------------------

In this section, we will write a script to run a reconstruction on a dataset collected from our benchtop optical ptychography playground. This mirrors very closely the reconstruction examples, however I encourage everyone to follow along, writing this script out line-by-line, to help you learn more permanently the process of writing a custom reconstruction script.

Our first step will be creating the file and filling out the boilerplate: All the imports we'll need.

.. code-block:: python

   from __future__ import division, print_function, absolute_import

   import CDTools
   from matplotlib import pyplot as plt
   import pickle


You can always import more libraries, like numpy, or pytorch, or pandas, or what have you, as needed. Next, we load the dataset and give it a look-over


.. code-block:: python

   filename = 'example_data/lab_ptycho_data.cxi'
   dataset = CDTools.datasets.Ptycho2DDataset.from_cxi(filename)

   dataset.inspect()
   plt.show()

Now, run this script! You should see a window pop up, showing a nanomap of the integrated intensities at each scan point on one side, and an individual diffraction pattern on the other. You can then click around to make sure that everything is in order.

Now that we know we have the data loaded and it looks good, we can go ahead and comment out the dataset inspecting code, and move on to creating a model. It's usually a good idea to start by loading a standard :code:`FancyPtycho` model without any special changes, sending it to the GPU.

.. code-block:: python
		
   model = CDTools.models.FancyPtycho.from_dataset(dataset)
   model.to(device='cuda')
   dataset.get_as(device='cuda')


We then try a basic Adam reconstruction with this model, with no changes to the defaults, to see how it works. 

.. code-block:: python

   for i, loss in enumerate(model.Adam_optimize(50, dataset)):
       model.inspect(dataset)
       print(i,loss)

   model.compare()
   plt.show()


It is worth noting here exactly how this code is working. The reconstruction methods are actually returning generators. Generators in pythons are objects that work like lists, or tuples, but have to be read out one item at a time, from left to right. The catch is that, instead of just reading out objects from a list, they can run arbitrary code each time they are asked for the "next" item.

In CDTools, every reconstruction method will return a generator. Whenever the generator is asked for the next item, it runs a single epoch of the reconstructionalgorithm, and then returns the average loss over that epoch as that next item. This allows the execution of the reconstruction algorithm to pause once every epoch, allowing some time for the user to run a small snippet of code to inspect how the reconstruction is coming along.

From the end user perspective, all this means is: follow the format above, or more generally put the :code:`model.Adam_optimize(n, dataset)` call anywhere that you would feel comfortable putting a call to :code:`range(n)` - list comprehensions, for loops, etc.

Once we run this, we can take a look at the result. What we see is pretty good, but we can see that there are some issues with the reconstruction near the edge, and the probe itself seems to be larger than the "stage" on which we're reconstructing it. So, we can make two tweaks to this code in response. First, we increase the oversampling ratio, which doubles the size of the stage (this often can cause other issues as well, but generally works well in situations like this where the probe is honestly too large.

.. code-block:: python

   model = CDTools.models.FancyPtycho.from_dataset(dataset, oversampling=2)


And secondly, we note that there don't seem to be any errors with the positioning. So we can just not reconstruct the probe positions, knowing that the initial guesses are already accurate enough. We can do this by writing the following line, just before we run the reconstruction for loop.

.. code-block:: python
		
   model.translation_offsets.requires_grad = False

What is going on here is that, when running the optimization algorithm, pytorch will automatically calculate gradients for and then optimize over a number of parameters defined in the model - this includes parameters like :code:`model.probe`, :code:`model.obj`, :code:`model.background`, etc. We can tell pytorch to stop calculating gradients for (and stop updating) any of these parameters by setting their :code:`requires_grad` property to :code:`False`.

After running this reconstruction, we can see that we're getting a little improvement (and a larger field of view) by using oversampling, but out in the corners we're nucleating extra probes! We can fix this by adding a probe support - that is, declating that the probe has to be defined only within a certain box. This can be done most easily with an argument to the model constructor:

.. code-block:: python
   
   model = CDTools.models.FancyPtycho.from_dataset(dataset, oversampling=2,
                                                   probe_support_radius=90)


It also seems like we need a few more iterations to finish converging, so we up the iteration count to 100.

.. code-block:: python

   for i, loss in enumerate(model.Adam_optimize(100, dataset)):
						   

Now we expect to get a nice reconstruction, so we can save the data. You can save the data in any form you like, once the relevant information is extracted from the model and put into a dictionary. The standard method for saving out this information is as follows:

.. code-block:: python

   with open('example_reconstructions/lab_ptycho.pickle', 'wb') as f:
       pickle.dump(model.save_results(dataset),f)

This is usually placed before the call to :code:`plt.show()`, to make sure that if the user manually exits the program once all the plot windows are opened, the data will still have been saved.

Now, your file should match the example file in examples/lab_ptycho_data.py. 


Datasets
--------

In this section, we will write a dataset class that can be used to import a nonstandard kind of ptychographic data where the beam is scanned longitudinally though the sample rather than rastered across the sample laterally. At the end of this tutorial, we will have written the class defined in 


Models
------

In this section, we will write a model to perform a reconstruction on the axial scanning ptychography d

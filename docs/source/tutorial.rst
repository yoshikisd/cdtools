Tutorial
========

This tutorial builds on the examples, leading to a more complete understanding of how to use CDTools and - importantly - how to extend it. First, we will cover the details of writing a useful reconstruction script for a particular experiment. Next, we will discuss how to implement a new dataset type for different kinds of coherent diffraction. Finally, we will go over how to make new models to cover specific types of ptychography which aren't described by any of the built-in models.


Reconstruction Scripts
----------------------

In this section, we will write a script to run a reconstruction on a dataset collected from our benchtop optical ptychography playground. This mirrors very closely the reconstruction examples, however I encourage everyone to follow along, writing this script out line-by-line, to help you learn more permanently the process of writing a custom reconstruction script.

Our first step will be creating the file and filling out the boilerplate: All the imports we'll need.

.. code-block:: python

   import cdtools
   from matplotlib import pyplot as plt
   from scipy import io


You can always import more libraries, like numpy, or pytorch, or pandas, or what have you, as needed. Next, we load the dataset and give it a look-over


.. code-block:: python

   filename = 'example_data/lab_ptycho_data.cxi'
   dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

   dataset.inspect()
   plt.show()

Now, run this script! You should see a window pop up, showing a nanomap of the integrated intensities at each scan point on one side, and an individual diffraction pattern on the other. You can then click around to make sure that everything is in order.

Now that we know we have the data loaded and it looks good, we can go ahead and comment out the dataset inspecting code, and move on to creating a model. It's usually a good idea to start by loading a standard :code:`FancyPtycho` model without any special changes, sending it to the GPU.

.. code-block:: python
		
   model = cdtools.models.FancyPtycho.from_dataset(dataset)
   model.to(device='cuda')
   dataset.get_as(device='cuda')


We then try a basic Adam reconstruction with this model, with no changes to the defaults, to see how it works. 

.. code-block:: python

   for loss in model.Adam_optimize(50, dataset):
       model.inspect(dataset)
       print(model.report())

   model.compare(dataset)
   plt.show()


It is worth noting here exactly how this code is working. The reconstruction methods are actually returning generators. Generators in pythons are objects that work like lists, or tuples, but have to be read out one item at a time, from left to right. The catch is that, instead of just reading out objects from a list, they can run arbitrary code each time they are asked for the "next" item.

In CDTools, every reconstruction method will return a generator. Whenever the generator is asked for the next item, it runs a single epoch of the reconstructionalgorithm, and then returns the average loss over that epoch as that next item. This allows the execution of the reconstruction algorithm to pause once every epoch, allowing some time for the user to run a small snippet of code to inspect how the reconstruction is coming along.

From the end user perspective, all this means is: follow the format above, or more generally put the :code:`model.Adam_optimize(n, dataset)` call anywhere that you would feel comfortable putting a call to :code:`range(n)` - list comprehensions, for loops, etc. In this case, we have called a function to plot out the current state of the reconstruction, and a function to print out the current loss and iteration time.

Once we run this, we can take a look at the result. What we see is pretty good, but we can see that there are some issues with the reconstruction near the edge, and the probe itself seems to be larger than the "stage" on which we're reconstructing it. So, we can make two tweaks to this code in response. First, we increase the oversampling ratio, which doubles the size of the stage (this often can cause other issues as well, but generally works well in situations like this where the probe is honestly too large.

.. code-block:: python

   model = cdtools.models.FancyPtycho.from_dataset(dataset, oversampling=2)


And secondly, we note that there don't seem to be any errors with the positioning. So we can just not reconstruct the probe positions, knowing that the initial guesses are already accurate enough. We can do this by writing the following line, just before we run the reconstruction for loop.

.. code-block:: python
		
   model.translation_offsets.requires_grad = False

What is going on here is that, when running the optimization algorithm, pytorch will automatically calculate gradients for and then optimize over a number of parameters defined in the model - this includes parameters like :code:`model.probe`, :code:`model.obj`, :code:`model.background`, etc. We can tell pytorch to stop calculating gradients for (and stop updating) any of these parameters by setting their :code:`requires_grad` property to :code:`False`.

After running this reconstruction, we can see that we're getting a little improvement (and a larger field of view) by using oversampling, but out in the corners we're nucleating extra probes! We can fix this by adding a probe support - that is, declating that the probe has to be defined only within a certain box. This can be done most easily with an argument to the model constructor:

.. code-block:: python
   
   model = cdtools.models.FancyPtycho.from_dataset(dataset, oversampling=2,
                                                   probe_support_radius=90)


It also seems like we need a few more iterations to finish converging, so we up the iteration count to 100.

.. code-block:: python

   for i, loss in enumerate(model.Adam_optimize(100, dataset)):
						   

Now we expect to get a nice reconstruction, so we can save the data. You can save the data in any form you like, once the relevant information is extracted from the model and put into a dictionary. The standard method for saving out this information is as follows:

.. code-block:: python

   io.savemat('example_reconstructions/lab_ptycho.pickle',
              model.save_results(dataset))

This is usually placed before the call to :code:`plt.show()`, to make sure that if the user manually exits the program once all the plot windows are opened, the data will still have been saved.

Now, your file should match the example file in examples/lab_ptycho_data.py. 


Datasets
--------

In this section, we will write a bare-bones dataset class for 2D ptychography data to demonstrate the process of writing a new dataset class. At the end of the tutorial, we will have written the file examples/basic_ptycho_dataset.py, which can be consulted for reference.

Basic Idea
++++++++++

At it's core, a dataset object for CDTools is just an object that implements the dataset interface from pytorch. For this reason, the base class (:code:`CDataset`) from which all the datasets are defined is itself a subclass of :code:`torch.utils.data.Dataset`. In addition, CDataset implements an extra layer that allows for a separation between the device (CPU or GPU) that the data is stored on and the device that it returns data on. This allows for GPU-based reconstructions on datasets that are too large to fit into the GPU in their entirety.

The pytorch Dataset interface is very simple. A dataset simply has to define two functions, :code:`__len__()` and :code:`__getitem__()`. Thus, we can always access the data in a Dataset :code:`mydata` using the syntax :code:`mydata[index]` or :code:`mydata[slice]`. Overriding these functions will be the first task in defining a new dataset.

In CDTools datasets, the layer that allows for separation between the device that data is stored on and the device that data is loaded onto is implemented in the :code:`__getitem__()` function. Instead of overriding this function directly, one should override the :code:`_load()` function, which is used internally by :code:`__getitem__()`.

In addition to acting as a pytorch Dataset, CDTools Datasets also work as interfaces to .cxi files. Therefore, when writing a new dataset, it is important to also override the functions :code:`to_cxi()` and :code:`from_cxi()` which handle writing to and reading from cxi files, respectively.

The final piece of the puzzle is the :code:`inspect()` method. This is not required to be defined for all datasets, however it is extremely valuable to offer a simple way of exploring a dataset visually. Therefore it is highly recommended to implement this function, which should load a plot or interactive plot that allows a user to visualize the data that they have loaded.

Writing the Skeleton
++++++++++++++++++++

We can start with the basic skeleton for this file. In addition to our standard imports, we also import the base CDataset class and the data tools. We then define an :code:`__all__` list as good practice, and set up the inheritance of our class.

.. code-block:: python

  
    import numpy as np
    import torch as t
    from matplotlib import pyplot as plt
    from cdtools.datasets import CDataset
    from cdtools.tools import data as cdtdata

    __all__ = ['BasicPtychoDataset']

    class BasicPtychoDataset(CDataset):
        """The standard dataset for a 2D ptychography scan"""
        pass


Initialization
++++++++++++++

The next thing to implement is the initialization code. Here we can leverage some of the work already done in the base CDataset class. There are a number of kinds of metadata that can be stored in a .cxi file that aren't related to the kind of experiment you're performing - sample ID, start and end times, and so on. The CDataset's initialization routine handles loading and storing these various kinds of metadata, so we can start the definition of our initialization routine by leveraging this:

.. code-block:: python

    def __init__(self, *args, **kwargs):
        super(BasicPtychoDataset,self).__init__(*args, **kwargs)


Of course, there is also some data that are unique to this kind of dataset. In this case, those data are the probe translations and the measured diffraction patterns. Therefore, we extend this definition to the following:

.. code-block:: python

    def __init__(self, translations, patterns, *args, **kwargs):
        """Initialize the dataset from python objects"""

        super(BasicPtychoDataset,self).__init__(*args, **kwargs)
        self.translations = t.Tensor(translations).clone()
        self.patterns = t.Tensor(patterns).clone()


Dataset Interface
+++++++++++++++++

The next set of functions to write are those that plug into the dataset interface. We want :code:`len(dataset)` to return the number of diffraction patterns, which is straightforward to implement.

For the :code:`_load()` implementation, we need to consider what format the data should be returned in. The standard for all CDTools datasets is to return a tuple of (inputs, output). The inputs should always be defined as a tuple of inputs, even if there is only one input for this kind of data. As we will see later in the section on constructing models, this makes it possible to write the automatic differentiation code in a way that is applicable to every model.

In this case, our "inputs" will be a tuple of (pattern index, probe translation). This is not the only reasonable choice - it would also be possible, for example to define the input as just a pattern index (and store the probe translations in the model). For simple ptychography models with no error correction, it's also possible to just take a probe translation as an input with no index. Requiring both is the compromise that's been implemented in the default ptychography models defined with CDTools, and therefore we will follow that format here.

.. code-block:: python

    def __len__(self):
        return self.patterns.shape[0]

    def _load(self, index):
        return (index, self.translations[index]), self.patterns[index]

Remember that it's not needed to worry about what device or datatype the data is stored as here, as the relevant conversions will be performed by the :code:`__getitem()` method defined in the superclass. However, we do generally implement a method, :code:`to()`, that moves the data back and forth between devices and datatypes. This lets a user speed up data loading onto the GPU by preloading the data, for example - provided there is enough space.

.. code-block:: python

    def to(self, *args, **kwargs):
        """Sends the relevant data to the given device and dtype"""
        super(BasicPtychoDataset,self).to(*args,**kwargs)
        self.translations = self.translations.to(*args, **kwargs)
        self.patterns = self.patterns.to(*args, **kwargs)

Here we can see that we first make sure to call the superclass function to handle sending any information (such as a pixel mask, or detector background) that would have been defined in CDataset to the relevant device. Then we handle the new objects that are defined specifically for this kind of dataset.


Loading and Saving
++++++++++++++++++

Now we turn to writing the tools to load and save data. First, to load the data, we override :code:`from_cxi()`, which is a factory method. In this case, we start by using the superclass to load the metadata. Then we explicitly load in and add the data that's specific to this dataset class

.. code-block:: python

    @classmethod
    def from_cxi(cls, cxi_file):
        """Generates a new CDataset from a .cxi file directly"""

        # Generate a base dataset
        dataset = CDataset.from_cxi(cxi_file)
        # Mutate the class to this subclass (BasicPtychoDataset)
        dataset.__class__ = cls

        # Load the data that is only relevant for this class
        patterns, axes = cdtdata.get_data(cxi_file)
        translations = cdtdata.get_ptycho_translations(cxi_file)

        # And now re-add it
        dataset.translations = t.Tensor(translations).clone()
        dataset.patterns = t.Tensor(patterns).clone()

        return dataset


Now to save the data, we override :code:`to_cxi()`, in a fairly self-explanatory way.

.. code-block:: python

    def to_cxi(self, cxi_file):
        """Saves out a BasicPtychoDataset as a .cxi file"""

        super(BasicPtychoDataset,self).to_cxi(cxi_file)
        cdtdata.add_data(cxi_file, self.patterns, axes=self.axes)
        cdtdata.add_ptycho_translations(cxi_file, self.translations)

Note that these functions should be defined to work on h5py file objects representing the .cxi files (.cxi files are just .h5 files with a special formatting).


Inspecting
++++++++++

The final piece of the puzzle is writing a function to look at your data! This is an important thing to work on for a dataset class that you intend to use regularly, as being able to easily peruse your raw data has incalculable value. Here, we satisfy ourselves with just plotting a random diffraction pattern.

.. code-block:: python

    def inspect(self):
        """Plots a random diffraction pattern"""

        index = np.random.randint(len(self))
        plt.figure()
        plt.imshow(self.patterns[index,:,:].cpu().numpy())


Notes
+++++

This is a bare-bones class, set up to demonstrate the minimum neccessary to develop a new type of dataset class. As a result, it doesn't implement a number of things that are useful or valuable in practice (and which the default Ptycho2DDataset does implement). That includes a useful data inspector, the ability to load datasets directly from filenames, and default tweaks to how metadata such as backgrounds and masks are loaded.

	
Models
------

In this section, we will write a basic model for 2D ptychography reconstructions. At the end of this tutorial, we will have written the class defined in examples/simple_ptycho_model.py


Basic Idea
++++++++++

Just like CDTools Datasets subclass pytorch Datasets, CDTools models subclass pytorch modules (yes, I know they are different words - we use the word "model" in CDTools to conform to usage in the world of ptychography/CDI). The major difference is that the base CDTools models also contains a few standard methods to run automatic differentiation reconstructions on itself. This isn't necessarily the cleanest or most portable approach, but we've found that it feels very natural from the perspective of an end user interacting with the toolbox only through some basic reconstruction scripts.

The models themselves have a :code:`model.forward()` function which contains the real meat. In any CDTools model, this forward function takes in a set of parameters describing the specific diffraction pattern to simulate, and outputs a simulated diffraction pattern. The inputs could be as simple as a diffraction pattern index, or could explicitly include other information like the probe position.

In practice, the forward model is defined in the top level :code:`CDIModel` class from which all other models are derived. The definition is quite simple:

.. code-block:: python
		
    def forward(self, *args):
        return self.measurement(self.forward_propagator(self.interaction(*args)))

So we can see that to fully implement this forward model, we have to define the three functions :code:`model.interaction()`, :code:`model.forward_propagator()`, and :code:`model.measurement()`, which simulate conceptual stages in the diffraction process.

Beyond the basic model definition, a few other tools need to be defined. The model has to be able to create itself from a dataset, has to have a loss function defined for use with automtic differentiation, has to know how to plot out it's progress, and has to be able to save out the results of a reconstruction. The details of how to implement all of this in a model are shown below.


Writing the Skeleton
++++++++++++++++++++

Once again, we start with the basic skeleton

.. code-block:: python

    import numpy as np
    import torch as t
    from cdtools.models import CDIModel
    from cdtools import tools

    __all__ = ['SimplePtycho']

    class SimplePtycho(CDIModel):
        """A simple ptychography model for exploring ideas and extensions"""
        pass

Note that we imported the full tools package, as we will find ourselves using many low-level functions defined there to implement the model.


Initialization from Python
++++++++++++++++++++++++++

Two initialization functions need to be written. First, we write the :code:`__init__()` function, which initializes the model from a collection of python objects describing the system. We then write an initializer that creates a model using a dataset to define the various parameters.

It's important to note that there's not requirement for what the arguments to the initialization function of any particular model should be, only that they contain enough information to run the simulations! It should be chosen in a model-by-model way to allow for the most transparent code.

.. code-block:: python

    def __init__(self, probe_basis, probe_guess, obj_guess,
                 min_translation = [0,0]):

        # We have to initialize the Module
        super(SimplePtycho,self).__init__()
	
        # We first save the relevant information
        self.min_translation = t.Tensor(min_translation)
        self.probe_basis = t.Tensor(probe_basis)

        # We rescale the probe so it learns at the same rate as the object
        self.probe_norm = t.max(t.abs(probe_guess).to(t.float32))
        self.probe = t.nn.Parameter(probe_guess.to(t.complex64)
                                    / self.probe_norm)
        self.obj = t.nn.Parameter(obj_guess.to(t.complex64))
		

Here, we chose to define the model based on a basis matrix describing the probe array, an initial guess at the probe, and an initial object. In addition, an optional offset for the translations is included. 

The first part of this initialization is quite straightforward - we create some tensors for the minimum translation and the probe basis. But then, the next two pieces of information that we save are defined as :code:`t.nn.Parameter` objects, not Tensors! Parameters are different from Tensors in two ways.

The first way that they are different is that, by default, they have the :code:`requires_grad` flag set to :code:`True`, which means that the information needed for gradient calculations will be stored on every Tensor that results from a calculation including a Parameter. The second difference is that, when a Parameter is added to a Module, the Module adds that parameter to a list, which can be accessed by calling :code:`module.parameters()`.

The key here is that the model itself is subclassing a pytorch Module. So, every parameter that we will attempt to reconstruct, we add to the CDTools model as a Parameter. This way, the model automatically knows which variables to update with gradient descent, and which to keep as they are. Here, we only need to reconstruct the probe and object.

One final note is that we actually store a scaled version of the probe. This is a hack. It is simply because the Adam optimization method, which is the most commonly used, uses learning rates that scale with the amplitude of the parameter, rather than with the amplitude of it's gradients. Unti pytorch implements separate parameter-group learning rates in Adam, rescaling all the parameters to have a typical amplitude near 1 is the best way to get well-behaved reconstructions.


Initialization from Dataset
+++++++++++++++++++++++++++

To initialize the object from a dataset, we need to start by extracting the relevant information from the dataset. Then we can simply call the constructor we defined earlier.

.. code-block:: python
		
    @classmethod
    def from_dataset(cls, dataset):
        # First, load the information from the dataset
        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']
	(indices, translations), patterns = dataset[:]

	# Then, generate the probe geometry
        ewg = tools.initializers.exit_wave_geometry
        probe_basis, probe_shape, det_slice =  ewg(det_basis,
                                                   det_shape,
                                                   wavelength,
                                                   distance,
						   opt_for_fft=False)

        # Next generate the object geometry
        pix_translations = tools.interactions.translations_to_pixel(
            probe_basis, translations)
        obj_size, min_translation = tools.initializers.calc_object_setup(
            probe_shape, pix_translations)

        # Finally, initialize the probe and  object using this information
        probe = tools.initializers.SHARP_style_probe(dataset,
                                                     probe_shape,
                                                     det_slice)

        obj = t.ones(obj_size, dtype=t.complex64)

        return cls(probe_basis, probe, obj, min_translation=min_translation)
    

Here, we start by pulling the basic geometric information from the dataset. Then, we use a number of the basic tools to do calculations such as finding the probe basis from the detector geometry, or calculating how big our object array should be.

Once we have the basic setup ready, we then use one of the initialization functions - in this case, :code:`tools.initializers.SHARP_style_probe`, to find a sensible initialization for the probe. This particular initialization is based on the approach used in the SHARP package. Once all the needed information has been collected, we initialize the object.


The Forward Model
+++++++++++++++++

First, we have to implement the interaction model. This function should take the inputs defining the specific diffraction pattern or collection of patterns, and return an exit wave or set of exit waves that would be expected from that set of inputs. This models the interaction of the probe with the sample

.. code-block:: python

    def interaction(self, index, translations):
        pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                             translations)
        pix_trans -= self.min_translation
        return tools.interactions.ptycho_2D_round(self.probe_norm * self.probe,
                                                  self.obj,
                                                  pix_trans)


Here, we take input in the form of an index and a translation. Note that this has to match the format that is output by the associated datasets that we will use for reconstruction.

We start by mapping the translation, given in real space, into pixel coordinates. Then, we use an "off-the-shelf" interaction model - :code:`ptycho_2d_round`, which models a standard 2D ptychography interaction, but rounds the translations to the nearest whole pixel (does not attempt subpixel translations).

The next three definitions amount to just choosing an off-the-shelf function to simulate each step in the chain.

.. code-block:: python

    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)

    def measurement(self, wavefields):
        return tools.measurements.intensity(wavefields)

    def loss(self, sim_data, real_data):
        return tools.losses.amplitude_mse(real_data, sim_data)


The forward propagator maps the exit wave to the wave at the surface of the detector, here using a far-field propagator. The measurement maps that exit wave to a measured pixel value, and the loss defines a loss function to attempt to minimize. The loss function we've chosen - the amplitude mean squared error - is the most broadly applicable one.


Device Management
+++++++++++++++++

Next we need to implement a :code:`to()` function, just like we did in the dataset, to allow the entire model to be moved between the GPU and CPU.

.. code-block:: python

    def to(self, *args, **kwargs):
        super(SimplePtycho, self).to(*args, **kwargs)
        self.min_translation = self.min_translation.to(*args,**kwargs)
        self.probe_basis = self.probe_basis.to(*args,**kwargs)
        self.probe_norm = self.probe_norm.to(*args,**kwargs)

We start by calling the same function for the superclass, which will take care of moving every parameter attached to the model. Then we manually take care of moving over the other variables which, while not being updated in the gradient descent, still need to be moved over to the new device!


Plotting
++++++++

The base CDIModel class has a function, :code:`model.inspect()`, which looks for a class variable called :code:`plot_list` and plots everything contained within. The plot list should be formatted as a list of tuples, with each tuple containing:

* The title of the plot
* A function that takes in the model and generates the relevant plot
* Optional, a function that takes in the model and returns whether or not the plot should be generated
  
.. code-block:: python

    plot_list = [
        ('Probe Amplitude',
         lambda self, fig: tools.plotting.plot_amplitude(self.probe, fig=fig, basis=self.probe_basis)),
        ('Probe Phase',
         lambda self, fig: tools.plotting.plot_phase(self.probe, fig=fig, basis=self.probe_basis)),
        ('Object Amplitude',
         lambda self, fig: tools.plotting.plot_amplitude(self.obj, fig=fig, basis=self.probe_basis)),
        ('Object Phase',
         lambda self, fig: tools.plotting.plot_phase(self.obj, fig=fig, basis=self.probe_basis))
    ]


In this case, we've made use of the convenience plotting functions defined in :code:`tools.plotting`.


Saving
++++++

At the moment, there is no consistent way to save out the results across the board. However, a function :code:`save_results()` should be defined, which should save out the results of the reconstruction into some reasonably formatted python object. Here we return a dictionary with the probe and object:

.. code-block:: python

    def save_results(self):
        probe = self.probe.detach().cpu().numpy()
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = self.obj.detach().cpu().numpy()
        return {'probe':probe,'obj':obj}


Testing
+++++++

We can test this model with a simple script, shown below. By filling in the backend here, we've been able to create a ptychography model that can be accessed and used in reconstructions via the same interface as the models we discussed in the examples section.

.. code-block:: python

    from basic_ptycho_dataset import BasicPtychoDataset
    from h5py import File
    from matplotlib import pyplot as plt
    
    filename = 'example_data/lab_ptycho_data.cxi'
    with File(filename, 'r') as f:
        dataset = BasicPtychoDataset.from_cxi(f)

        
    model = SimplePtycho.from_dataset(dataset)
    
    model.to(device='cuda')
    dataset.get_as(device='cuda')

    for loss in model.Adam_optimize(100, dataset):
        model.inspect(dataset)
        print(model.report())
    
    model.compare(dataset)
    plt.show()


Happy modeling!

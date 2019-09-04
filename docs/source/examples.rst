Examples
========

Included with the repository are a number of example scripts that demonstrate how various aspects of CDTools work. It is recommended to read through several of them before continuing to the tutorial, to get a feel for how the ptychography scripts work.


Simple Ptycho
-------------

This script runs a ptychography reconstruction using the SimplePtycho model. This model implements a straightforward transmission ptychography geometry, with no position refinement, incoherent modes, or other doodads (only a background model). 

.. literalinclude:: ../../examples/simple_ptycho.py

Running this script leads to a very poor reconstruction on the example data, however depending on the dataset the model can produce reasonable results. When reading this script, note the basic workflow: First, data is read into a dataset object. Then, a model is created to match the geometry stored in the dataset, with a default initialization for all the parameters. Next, a reconstruction is run (and the progress reported out). Finally, the finished reconstruction is inspected.


Gold Balls Ptycho
-----------------


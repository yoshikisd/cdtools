"""This module contains all the models for different CDI Reconstructions

All the reconstructions are coordinated through the ptychography models
defined here. The models are, at their core, just subclasses of the 
:code:`torch.nn.model` class, so they contain the same structure of
parameters, etc. Their central functionality is as a simulation that maps
some input (usually, the index number of a scan point) to an output that
corresponds to the measured data (usually, a diffraction pattern). This
model can then be used as the heart of an automatic differentiation
reconstruction which retrieves the parameters that were used in the model.

A main CDIModel class is defined in the base.py file, and models for
various CDI geometries can be defined as subclasses of this base model.
The subclasses of the main CDIModel class are required to implement a set of
functions defined in the base.py file. Example implementations of
these functions can be found in the code for the SimplePtycho class.

Finally, it is recommended to read through the tutorial section on
defining a new ptychography model before attempting to do so.

"""

from __future__ import division, print_function, absolute_import

# I don't believe that __all__ really needed, but it's nice to define it
# to be explicit that import * is safe
__all__ = ['CDIModel', 'SimplePtycho', 'FancyPtycho', 'Bragg2DPtycho', 'SMatrixPtycho', 'RPI']

from CDTools.models.base import CDIModel
from CDTools.models.simple_ptycho import SimplePtycho
from CDTools.models.fancy_ptycho import FancyPtycho
from CDTools.models.pinhole_plane_ptycho import PinholePlanePtycho
from CDTools.models.bragg_2d_ptycho import Bragg2DPtycho
from CDTools.models.s_matrix_ptycho import SMatrixPtycho
from CDTools.models.multislice_2d_ptycho import Multislice2DPtycho
from CDTools.models.rpi import RPI

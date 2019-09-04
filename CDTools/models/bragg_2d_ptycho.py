from __future__ import division, print_function, absolute_import

import torch as t
from CDTools.models import CDIModel
from CDTools.datasets import Ptycho_2D_Dataset
from CDTools import tools
from CDTools.tools import cmath
from CDTools.tools import plotting as p
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from copy import copy


#
# Key ideas:
# 1) To a first approximation, do the reconstruction on a parallelogram
#    shaped grid on the sample which is conjugate to the detector coordinates
# 2) Simulate a probe in those same coordinates, but propagate it back and
#    forth (along with the translations), using the angular spectrum method
# 3) Apply a correction to the simulated data to account for the tilt of the
#    sample with respect to the detector
# 4) Include a correction for the thickness of the sample
#


class Bragg2DPtycho(CDIModel):

    def __init__(self, wavelength, detector_geometry,
                 probe_basis, probe_guess, obj_guess,
                 detector_slice=None,
                 surface_normal=np.array([0.,0.,1.]),
                 min_translation = t.Tensor([0,0]),
                 background = None, translation_offsets=None, mask=None,
                 weights = None, translation_scale = 1, saturation=None,
                 probe_support = None, obj_support=None, oversampling=1):

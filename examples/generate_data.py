from __future__ import division, print_function, absolute_import

import CDTools
from CDTools.tools.initializers import gaussian
from CDTools.tools.propagators import far_field
from CDTools.tools.measurements import intensity

from CDTools.tools.cmath import *
from CDTools.tools import interactions
import h5py
import numpy as np
from matplotlib import pyplot as plt
import torch as t
import scipy.misc



def simulate_pattern(pixel_translation_step = (10, 10), probe = None, intensity_err = True,
                    random_noise = True, translation_err = True, background_noise = True):
    """
    Generates diffraction data from scipy's face image with a gaussian phase.
    Can simulate different experimental errors, including:
        * Varying probe intensities over time
        * Nanopositioner location uncertainties
        * Background detector noise
        * Random noise
        * Uncentered probe

    We can do this by simulating the data collection process with a large
    probe array, and changing the translation by a random (integer) amount.
    Then, we can scale the intensity by a random amount (close to 1),
    replace pixels with some scaled version of their current value to simulate
    random noise, then add in horizontal and vertical detector background noise.
    Finally, we slice the simulated diffraction patterns to uncenter the probe
    on the final patterns [TO DO].

    Args:
        pixel_translation_step (array-like) : a tuple containing (y pixel translation, x pixel translation)
        probe (np.ndarray) : a two-dimensional array representing the probe wavefunction. Defaults to a gaussian probe
        intensity_err (Boolean) : Defaults to True, to simulate varying probe intensities over time
        random_noise (Boolean) : Defaults to True, indicating adding random noise to the diffraction patterns
        translation_err (Boolean) : Defaults to True, to simulate nanopositioner errors
        background_noise (Boolean) : [TO DO]
    Returns:
        patterns (t.Tensor) : NxMx2 array of simulated diffraction patterns with error
        translations (t.Tensor) : N*Mx2 array of instrument translations
    """
    obj = scipy.misc.face()
    # Convert to grayscale
    obj = 0.2989 * obj[:,:,0] + 0.5870 * obj[:,:,1] + 0.1140 * obj[:,:,2]
    phase = gaussian(obj.shape, [500,500], amplitude=1, center = None)
    obj = obj*np.exp(1j*torch_to_complex(phase))

    if probe is None:
        probe = torch_to_complex(gaussian([512,512], [5,5], amplitude=1, center = None))

    obj_shape = np.array(obj.shape)
    probe_shape = np.array(probe.shape)

    translation_range = (obj_shape-probe_shape)//np.array(pixel_translation_step)*np.array(pixel_translation_step)
    translations = np.mgrid[0:translation_range[0]+pixel_translation_step[0]:pixel_translation_step[0], 0:translation_range[1]+pixel_translation_step[1]:pixel_translation_step[1]]
    translations = translations.T.reshape((translations.T.shape[0]*translations.T.shape[1], 2))
    translations = t.tensor(translations, dtype = t.float32)
    ideal_translations = t.tensor(translations, dtype = t.float32)
    obj = complex_to_torch(obj)
    probe = complex_to_torch(probe)

    if translation_err:
        err = t.normal(mean=t.zeros(translations.shape), std=t.ones(translations.shape)/2)
        translations += err
        # Ensure that the new translations don't go off the object area
        translations[translations<0] = 0
        translation_range = t.tensor(translation_range, dtype = t.float32)
        translations[...,0][translations[...,0] > translation_range[0]] = translation_range[0]
        translations[...,1][translations[...,1] > translation_range[1]] = translation_range[1]

    patterns = intensity(far_field(interactions.ptycho_2D_round(probe, obj, translations)))

    if intensity_err:
        patterns *= t.normal(mean = t.ones(patterns.shape), std = t.ones(patterns.shape)/32)

    if background_noise:
        patterns += t.normal(mean = t.zeeros(patterns.shape), std = t.ones(patterns.shape)*t.max(patterns)/32)


    return patterns, ideal_translations

from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
import torch as t
import numpy as np

#
# This file will host tools to turn a propagated wavefield into a measured
# intensity pattern on a detector
#

__all__ = ['intensity', 'incoherent sum', 'quadratic_background']


def intensity(wavefield, detector_slice=None, epsilon=1e-7, saturation=None):
    """Returns the intensity of a wavefield
    
    The intensity is defined as the magnitude squared of the
    wavefront. If a detector slice is given, the returned array
    will only include that slice from the simulated wavefront.
    
    Args:
        wavefield (torch.Tensor) : A JxMxNx2 stack of complex wavefields
        detector_slice (slice) : Optional, a slice or tuple of slices defining a section of the simulation to return
        saturation (float) : Optional, a maximum saturation value to clamp the resulting intensities to

    Returns:
        torch.Tensor : A real MxN array storing the wavefield's intensities
    """
    if detector_slice is None:
        output = cmath.cabssq(wavefield) + epsilon
    else:
        if wavefield.dim() == 3:
            output = cmath.cabssq(wavefield[detector_slice]) + epsilon
        else:
            output = cmath.cabssq(wavefield[(np.s_[:],) + detector_slice]) + epsilon
    if saturation is None:
        return output
    else:
        return t.clamp(output,0,saturation)


def incoherent_sum(wavefields, detector_slice=None, epsilon=1e-7, saturation=None):
    """Returns the incoherent sum of the intensities of the wavefields
    
    The intensity is defined as the sum of the magnitudes squared of
    the wavefields. If a detector slice is given, the returned array
    will only include that slice from the simulated wavefronts.

    The first index is the index of the diffraction pattern to measure,
    the second index is the set of incoherently adding patterns, and
    the next two indices index the wavefield. The final index is the complex
    index.
    
    Args:
        wavefields (torch.Tensor) : A JxLxMxNx2 stack of complex wavefields
        detector_slice (slice) : Optional, a slice or tuple of slices defining a section of the simulation to return
        saturation (float) : Optional, a maximum saturation value to clamp the resulting intensities to

    Returns:
        torch.Tensor : A real MxN array storing the incoherently summed intensities
    """
    # This syntax just adds an axis to the slice to preserve the J direction
    if detector_slice is None:
        output = t.sum(cmath.cabssq(wavefields),dim=-3) + epsilon
    else:
        if wavefields.dim() == 4:
            output =  t.sum(cmath.cabssq(wavefields[(np.s_[:],)+detector_slice]),dim=0) + epsilon
        else:
            output = t.sum(cmath.cabssq(wavefields[(np.s_[:],np.s_[:])+detector_slice]),dim=0) + epsilon
                 
    if saturation is None:
        return output
    else:
        return t.clamp(output,0,saturation)


def quadratic_background(wavefield, background, detector_slice=None, measurement=intensity, epsilon=1e-7, saturation=None):
    """Returns the intensity of a wavefield plus a background
    
    The intensity is calculated via the given measurment function 
    Then, the square of the given background is added. This kind
    of background model is commonly used to enforce positivity of the
    background model.

    Args:
        wavefield (torch.Tensor) : A JxMxNx2 stack of complex wavefields
        background (torch.Tensor) : An tensor storing the square root of the detector background
        detector_slice (slice) : Optional, a slice or tuple of slices defining a section of the simulation to return
        measurement (function) : Optional, the measurement function to use. The default is measurements.intensity
        saturation (float) : Optional, a maximum saturation value to clamp the resulting intensities to

    Returns:
        torch.Tensor : A real MxN array storing the wavefield's intensities
    """
    if detector_slice is None:
        output = measurement(wavefield, epsilon=epsilon) + background**2
    else:
        output = measurement(wavefield, detector_slice, epsilon=epsilon) \
            + background**2

    if saturation is None:
        return output
    else:
        return t.clamp(output,0,saturation)

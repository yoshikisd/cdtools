from __future__ import division, print_function, absolute_import
from CDTools.tools.cmath import *
import torch as t

__all__ = ['modulus', 'support']

def modulus(wavefront, intensities, mask = None):
    """Implements the modulus constraint in torch

    This accepts a torch tensor representing the propagated simulated wavefront(s),
    where the last dimension represents the real and imaginary components of
    the propagated wavefield(s). It projects the modulus of the diffraction pattern
    onto the modulus of the simulated wavefield.

    It assumes that the wavefront is stored in an array
    [i,j] where i corresponds to the y-axis and j corresponds to the
    x-axis, with the origin following the CS standard of being in the
    upper right.

    Args:
        wavefront (torch.Tensor) : The JxNxMx2 stack of complex propagated wavefronts
        intensities (torch.Tensor): The measured diffraction pattern(s) stored as an JxNxM stack of real tensors
        mask (torch.Tensor) : Mask for the intensities array with shape JxNxM, where bad detector pixels are set to 0 and usable pixels set to 1
    Returns:
        torch.Tensor : The JxNxMx2 propagated wavefield with corrected intensities
    """
    if mask is None:
        return cmult(cphase(wavefront), intensities**.5)
    else:
        return cmult(cphase(wavefront), intensities.masked_select(mask)**.5)


def support(wavefront, mask):
    """Implements the support constraint in torch

    This accepts a torch tensor representing the propagated simulated wavefront(s),
    where the last dimension represents the real and imaginary components of
    the propagated wavefield(s). It projects the support of the imaged object
    onto the simulated wavefront via a mask.

    It assumes that the wavefront is stored in an array
    [i,j] where i corresponds to the y-axis and j corresponds to the
    x-axis, with the origin following the CS standard of being in the
    upper right.

    Args:
        wavefront (torch.Tensor) : The JxNxMx2 stack of complex propagated wavefronts
        mask (torch.Tensor) : Mask for the intensities array with shape JxNxM, where bad detector pixels are set to 0 and usable pixels set to 1
    Returns:
        torch.Tensor : The JxNxMx2 wavefield with the mask applied
    """
    return wavefront.masked_select(mask)

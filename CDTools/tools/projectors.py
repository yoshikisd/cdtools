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
    # Calculate amplitudes from intensities
    amplitudes = t.sqrt(intensities)
    # Normalize wavefront so the complex elements have modulus one
    wavefront_mag = cabs(wavefront)
    projected = wavefront * (amplitudes / wavefront_mag)[...,None]
    # Replace amplitude of wavefront with measured amplitude
    if mask is not None:
        selection = mask == 0
        # Apply the mask to replace unmasked pixels in the original wavefront
        projected = projected.masked_scatter(selection, wavefront.masked_select(selection))

    return projected


def support(wavefront, support):
    """Implements the support constraint in torch

    This accepts a torch tensor representing (a) simulated wavefield(s),
    where the last dimension represents the real and imaginary components of
    the propagated wavefield(s). It projects the support of the imaged object
    onto the simulated wavefront via a support mask.

    It assumes that the wavefront is stored in an array
    [i,j] where i corresponds to the y-axis and j corresponds to the
    x-axis, with the origin following the CS standard of being in the
    upper right.

    Args:
        wavefront (torch.Tensor) : The JxNxMx2 stack of complex propagated wavefronts
        support (torch.Tensor) : An NxM support, with 1s within the support and 0s outside 
    
    Returns:
        torch.Tensor : The JxNxMx2 wavefield with the support mask applied
    """
    return wavefront * support.to(wavefront.dtype)[...,None]


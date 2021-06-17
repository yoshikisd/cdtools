"""This module contains tools to simulate various measurement models

All the measurements here are safe to use in an automatic differentiation
model. There exist tools to simulate detectors with finite saturation
thresholds, backgrounds, and more.
"""

from __future__ import division, print_function, absolute_import

import torch as t
import numpy as np
from torch.nn.functional import avg_pool2d

#
# This file will host tools to turn a propagated wavefield into a measured
# intensity pattern on a detector
#

__all__ = ['intensity', 'incoherent_sum', 'density_matrix',
           'quadratic_background']


def intensity(wavefield, detector_slice=None, epsilon=1e-7, saturation=None, oversampling=1):
    """Returns the intensity of a wavefield
    
    The intensity is defined as the magnitude squared of the
    wavefront. If a detector slice is given, the returned array
    will only include that slice from the simulated wavefront.
    
    Parameters
    ----------
    wavefield : torch.Tensor
        A JxMxNx2 stack of complex wavefields
    detector_slice : slice
        Optional, a slice or tuple of slices defining a section of the simulation to return
    saturation : float
        Optional, a maximum saturation value to clamp the resulting intensities to
    oversampling : int
        Default 1, the width of the region pixels in the wavefield to bin into a single detector pixel

    Returns
    -------
    sim_patterns : torch.Tensor
        A real MxN array storing the wavefield's intensities
    """
    output = t.abs(wavefield)**2
    
    # Now we apply oversampling
    if oversampling != 1:
        if wavefield.dim() == 2:
            output = avg_pool2d(output.unsqueeze(0), oversampling)[0]
        else:
            output = avg_pool2d(output, oversampling)

    # Then we grab the detector slice
    if detector_slice is not None:
        if wavefield.dim() == 2:
            output = output[detector_slice]
        else:
            output = output[(np.s_[:],) + detector_slice]

    # And now saturation    
    if saturation is None:
        return output + epsilon
    else:
        return t.clamp(output + epsilon,0,saturation)


def density_matrix(wavefields, density_matrix, detector_slice=None, epsilon=1e-7, saturation=None, oversampling=1):
    """Returns the intensities associated with a given density matrix state

    The essential idea is that the most general description of a light field
    at the detector plane will consist of a density matrix state. Here, that
    low rank density matrix state is encoded as a set of basis wavefields
    and a density matrix in that basis.

    For computational efficiency, the density matrix is coded in an unusual
    format. The density matrix formally is a complex Hermetian matrix,
    which also happens to be positive definite. Here, we store it as a
    real-valued matrix, where the upper triangle corresponds to the real
    part of the elements in the upper triangle, and the lower triangle
    corresponds to the imaginary parts. The elements on the diagonal are
    purely real, and are stored as they are.

    As with other multi-mode measurement functions, the modes are stored in
    the first index, and the index of the diffraction pattern in the stack
    of diffraction patterns is the second index. The stack-direction index
    can be omitted if only a single pattern needs to be simulated
    
    It is important to note that this method does not inforce the positive
    definiteness of the density matrix, this it is possible for negative
    values of intensity to appear if the underlying density matrix passed
    to this method is not positive definite

    Parameters
    ----------
    wavefields : torch.Tensor
        An Lx(Jx)MxNx2 stack of complex wavefields
    density_matrix : torch.Tensor
        A (Jx)LxL stack of real-valued representations of density matrices, as per above
    saturation : float
        Optional, a maximum saturation value to clamp the resulting intensities to
    oversampling : int
        Default 1, the width of the region pixels in the wavefield to bin into a single detector pixel

    Returns
    -------
    sim_patterns : torch.Tensor 
        A real Lx(Jx)MxN array storing the incoherently summed intensities

    """
    
    #if wavefields.dim() == 4:
    #    wavefields.unsqueeze(1)
    #    single_frame = True
    #elif wavefields.dim() == 5:
    #    single_frame=False
        
    output = t.zeros(wavefields.shape[1:-1],
                     dtype=wavefields.dtype,
                     device=wavefields.device)
    
    # flat is better than nested, but simple is better than complex...
    for (i,j) in ((i,j) for i in range(density_matrix.shape[-2])
                  for j in range(density_matrix.shape[-1])):
        if i == j: # diagonal
            output += density_matrix[...,i,j,None,None] \
                * t.abs(wavefields[i])**2
        if i < j: # upper triangle, real part 
            output += 2 * density_matrix[...,i,j,None,None] \
                * (wavefields[i,...,0] * wavefields[j,...,0]
                   +  wavefields[i,...,1] * wavefields[j,...,1])
        if i > j: # lower triangle, imaginary part
            # We pull the i,jth element from the density matrix,
            # but this correponds to wavefield j and wavefield i,
            # unlike above where it was wavefield i and j (swapped order).
            # We also get the one negative sign because this is the imaginary
            # part
            output += 2 * density_matrix[...,i,j,None,None] \
                * (wavefields[j,...,0] * wavefields[i,...,1]
                   -  wavefields[j,...,1] * wavefields[i,...,0])
        
    # Now we apply oversampling
    if oversampling != 1:
        if wavefields.dim() == 4:
            output = avg_pool2d(output.unsqueeze(0), oversampling)[0]
        else:
            output = avg_pool2d(output, oversampling)
            
    # Then we grab the detector slice
    if detector_slice is not None:
        if wavefields.dim() == 4:
            output = output[detector_slice]
        else:
            output = output[(np.s_[:],) + detector_slice]

    if saturation is None:
        return t.clamp(output,min=0) + epsilon
    else:
        return t.clamp(output + epsilon,0,saturation)
    
            
            
def incoherent_sum(wavefields, detector_slice=None, epsilon=1e-7, saturation=None, oversampling=1):
    """Returns the incoherent sum of the intensities of the wavefields
    
    The intensity is defined as the sum of the magnitudes squared of
    the wavefields. If a detector slice is given, the returned array
    will only include that slice from the simulated wavefronts.

    The (-4th) index is the set of incoherently adding patterns, and any
    indexes further to the front correspond to the set of diffraction patterns
    to meaasure. The (-3rd) and (-2nd) indices are the wavefield, and the final
    index is the complex index
    
    Parameters
    ----------
    wavefields : torch.Tensor
        An LxJxMxNx2 stack of complex wavefields
    detector_slice : slice
        Optional, a slice or tuple of slices defining a section of the simulation to return
    saturation : float
        Optional, a maximum saturation value to clamp the resulting intensities to
    oversampling : int
        Default 1, the width of the region pixels in the wavefield to bin into a single detector pixel

    Returns
    -------
    sim_patterns : torch.Tensor 
        A real LXMxN array storing the incoherently summed intensities
    """

    output = t.sum(t.abs(wavefields)**2,dim=-3) 

    # Now we apply oversampling
    if oversampling != 1:
        if wavefields.dim() == 4:
            output = avg_pool2d(output.unsqueeze(0), oversampling)[0]
        else:
            output = avg_pool2d(output, oversampling)
            
    # Then we grab the detector slice
    if detector_slice is not None:
        if wavefields.dim() == 4:
            output = output[detector_slice]
        else:
            output = output[(np.s_[:],) + detector_slice]

    if saturation is None:
        return output + epsilon
    else:
        return t.clamp(output + epsilon,0,saturation)


def quadratic_background(wavefield, background, *args, detector_slice=None, measurement=intensity, epsilon=1e-7, saturation=None, oversampling=1):
    """Returns the intensity of a wavefield plus a background
    
    The intensity is calculated via the given measurment function 
    Then, the square of the given background is added. This kind
    of background model is commonly used to enforce positivity of the
    background model.

    Parameters
    ----------
    wavefield : torch.Tensor
        A JxMxNx2 stack of complex wavefields
    background : torch.Tensor
        An tensor storing the square root of the detector background
    detector_slice : slice
        Optional, a slice or tuple of slices defining a section of the simulation to return
    measurement : function
        Default is measurements.intensity, the measurement function to use.
    saturation : float
        Optional, a maximum saturation value to clamp the resulting intensities to
    oversampling : int
        Default 1, the width of the region pixels in the wavefield to bin into a single detector pixel

    Returns
    -------
    sim_patterns : torch.Tensor
        A real MxN array storing the wavefield's intensities
    """
    
    if detector_slice is None:
        output = measurement(wavefield, *args, epsilon=epsilon,
                             oversampling=oversampling) + background**2
    else:
        output = measurement(wavefield, *args, detector_slice=detector_slice,
                             epsilon=epsilon, oversampling=oversampling) \
                             + background**2

    # This has to be done after the background is added, hence we replicate
    # it here
    if saturation is None:
        return output
    else:
        return t.clamp(output,0,saturation)

from __future__ import division, print_function, absolute_import

from CDTools.tools.cmath import *
import torch as t
from scipy import fftpack
import numpy as np

__all__ = ['far_field', 'near_field',
           'generate_angular_spectrum_propagator',
           'inverse_far_field', 'inverse_near_field']


def far_field(wavefront):
    """Implements a far-field propagator in torch

    This accepts a torch tensor, where the last dimension
    represents the real and imaginary components of the wavefield,
    and returns the far-field propagated version of it assuming it matches the
    detector dimensions. It assumes that the
    propagation is purely far-field, without checking that the geometry
    is consistent with that assumption.


    It also assumes that the real space wavefront is stored in an array
    [i,j] where i corresponds to the y-axis and j corresponds to the
    x-axis, with the origin following the CS standard of being in the
    upper right. The zero frequency component of the propagated wavefield is
    shifted to the center of the array.

    Args:
        wavefront (torch.Tensor) : The JxNxMx2 stack of complex wavefronts to be propagated
    Returns:
        torch.Tensor : The JxNxMx2 propagated wavefield
    """

    return fftshift(t.fft(ifftshift(wavefront), 2, normalized=True))


def inverse_far_field(wavefront):
    """Implements the inverse of the far-field propagator in torch

    This accepts a torch tensor, where the last dimension
    represents the real and imaginary components of the propagated wavefield,
    and returns the un-propagated array.

    It assumes that the real space wavefront is stored in an array
    [i,j] where i corresponds to the y-axis and j corresponds to the
    x-axis, with the origin following the CS standard of being in the
    upper right. The zero frequency component of the propagated wavefield is
    assumed to be the center of the array.

    Args:
        wavefront (torch.Tensor) : The JxNxMx2 stack of complex wavefronts propagated to the far-field
    Returns:
        torch.Tensor : The JxNxMx2 exit wavefield
    """
    return fftshift(t.ifft(ifftshift(wavefront), 2, normalized=True))


def generate_angular_spectrum_propagator(shape, spacing, wavelength, z, *args, **kwargs):
    """Generates an angular-spectrum based near-field propagator from experimental quantities

    This function generates an angular-spectrum based near field
    propagator that will work on torch Tensors. The function is structured
    this way - to generate the propagator first - because the
    generation of the propagation mask is a bit expensive and if this
    propagator is used in a reconstruction program, then it will be best
    to calculate this mask once and close over it.

    Formally, this propagator is the complex conjugate of the fourier
    transform of the convolution kernel for light propagation in free
    space

    Args:
        shape (iterable) : The shape of the arrays to be propagated
        spacing (iterable) : The pixel size in each dimension of the arrays to be propagated
        wavelength (float) : The wavelength of light to simulate propagation of
        z (float) : The distance to simulate propagation over

    Returns:
        torch.Tensor : A phase mask which accounts for the phase change that each plane wave will undergo.
    """

    ki = 2 * np.pi * fftpack.fftfreq(shape[0],spacing[0])
    kj = 2 * np.pi * fftpack.fftfreq(shape[1],spacing[1])
    Kj, Ki = np.meshgrid(kj,ki)

    # Define this as complex so the square root properly gives
    # k>k0 components imaginary frequencies    
    k0 = np.complex128((2*np.pi/wavelength))
    
    propagator = np.exp(1j*np.sqrt(k0**2 - Ki**2 - Kj**2) * z)

    # Take the conjugate explicitly here instead of negating
    # the previous expression to ensure that complex frequencies
    # get mapped to values <1 instead of >1
    propagator = complex_to_torch(np.conj(propagator)) 

    return propagator.to(*args, **kwargs)


def near_field(wavefront, angular_spectrum_propagator):
    """ Propagates a wavefront via the angular spectrum method

    This function accepts an 3D torch tensor, where the last dimension
    represents the real and imaginary components of the wavefield, and
    returns the near-field propagated version of it. It does this
    using the supplied angular spectrum propagator, which is a premade
    phase mask.


    Args:
        wavefront (torch.Tensor) : The JxNxMx2 stack of complex wavefronts to be propagated
        angular_spectrum_propagator (torch.Tensor) : The NxM phase mask to be applied during propagation

    Returns:
        torch.Tensor : The propagated wavefront 
    """
    return t.ifft(cmult(angular_spectrum_propagator,t.fft(wavefront,2)), 2)



def inverse_near_field(wavefront, angular_spectrum_propagator):
    """ Inverse ropagates a wavefront via the angular spectrum method

    This function accepts an 3D torch tensor, where the last dimension
    represents the real and imaginary components of the wavefield, and
    returns the near-field propagated version of it. It does this
    using the supplied angular spectrum propagator, which is a premade
    phase mask.

    It propagates the wave using the conjugate of the supplied phase mask,
    which corresponds to the inverse propagation problem.


    Args:
        wavefront (torch.Tensor) : The JxNxMx2 stack of complex wavefronts to be propagated
        angular_spectrum_propagator (torch.Tensor) : The NxM phase mask to be applied during propagation

    Returns:
        torch.Tensor : The inverse propagated wavefront
    """
    return t.ifft(cmult(t.fft(wavefront,2), cconj(angular_spectrum_propagator)), 2)




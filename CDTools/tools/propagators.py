"""This module contains various propagators for light fields

All the functions here are designed for use in an automatic differentiation
ptychography model. Each function implements a different propagator.
"""
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

    Parameters
    ----------
    wavefront : torch.Tensor
        The JxNxMx2 stack of complex wavefronts to be propagated
    
    Returns
    -------
    propagated : torch.Tensor
        The JxNxMx2 propagated wavefield
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

    Parameters
    ----------
    wavefront : torch.Tensor
        The JxNxMx2 stack of complex wavefronts propagated to the far-field
    
    Returns
    -------
    propagated : torch.Tensor
        The JxNxMx2 exit wavefield
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

    Parameters
    ----------
    shape : array
        The shape of the arrays to be propagated
    spacing : array
        The pixel size in each dimension of the arrays to be propagated
    wavelength : float
        The wavelength of light to simulate propagation of
    z : float
        The distance to simulate propagation over

    Returns
    -------
    propagator : torch.Tensor
        A phase mask which accounts for the phase change that each plane wave will undergo.
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


def generate_generalized_angular_spectrum_propagator(shape, basis, wavelength, offset_vector, *args, propagate_along_offset=True, **kwargs):
    """Generates an angular-spectrum based near-field propagator from experimental quantities

    This function generates an angular-spectrum based near field
    propagator that will work on torch Tensors. The function is structured
    this way - to generate the propagator first - because the
    generation of the propagation mask is a bit expensive and if this
    propagator is used in a reconstruction program, then it will be best
    to calculate this mask once and close over it.

    Formally, this propagator is the complex conjugate of the fourier
    transform of the convolution kernel for light propagation in free
    space. It will map a ligh field at an input plane, with the size
    and shape defined by the shape and basis inputs, and map it to a
    plane of the same size and shape offset by the offset vector. It
    is designed to work on any wavefield defined on an array of
    parallelograms.

    In addition, if propagate_along_offset is true, there is an assumed phase
    ramp applied to the wavefield before propagation, defined such that a
    feature with uniform phase will propagate along the direction of the 
    defined offset vector. This decision provides the best numerical
    stability and allows for the simple setup of light fields copropagating
    with the coordinate system.


    Parameters
    ----------
    shape : array
        The shape of the arrays to be propagated
    basis : array
        The (2x3) set of basis vectors describing the array to be propagated
    wavelength : float
        The wavelength of light to simulate propagation of
    propagation_vector : array
        The displacement to propagate the wavefield along.

    Returns
    -------
    propagator : torch.Tensor
        A phase mask which accounts for the phase change that each plane wave will undergo.
    """

    
    # First we calculate a dual basis for the real space grid
    inv_basis =  np.linalg.pinv(basis).transpose()

    # Then we calculate the frequencies in (i,j) space
    ki = 2 * np.pi * fftpack.fftfreq(shape[0])
    kj = 2 * np.pi * fftpack.fftfreq(shape[1])
    K_ij = np.stack(np.meshgrid(ki,kj, indexing='ij'))

    # Now we convert these to frequencies in reciprocal space
    # These frequencies span the 2D plane of the input wavefield.
    K_xyz = np.tensordot(inv_basis, K_ij, axes=1)

    # Now we need to apply two corrections to the standard AS method.
    # First, we calculate a phase mask which corresponds to the
    # shift of the final plane away from the perpendicular direction
    # from the input plane. We don't need to extract the perpendicular
    # component of the shift because the K_xyz vectors are naturally in the
    # input plane.

    # This may have a sign error - must be checked
    phase_mask = np.exp(1j * np.tensordot(offset_vector,K_xyz,axes=1))
    

    # Next, we apply a shift to the k-space vectors which sets up
    # propagation such that a uniform phase object will propagate along the
    # offset axis. This is not modeling a physical effect, but simply is
    # the clearest way to do a rigorous simulation while preventing
    # aliasing-related challenges. If used (as is by default), be aware
    # and prepare the input wavefields appropriately.
    perpendicular_dir = np.cross(basis[:,1],basis[:,0])
    perpendicular_dir /= np.linalg.norm(perpendicular_dir)
    offset_perpendicular = np.dot(perpendicular_dir, offset_vector)
    offset_parallel = offset_vector - perpendicular_dir * offset_perpendicular
    k0 = 2*np.pi/wavelength
    k_offset = offset_parallel * k0 / np.sqrt(offset_perpendicular**2 +
                                    np.linalg.norm(offset_parallel)**2)
    K_xyz = K_xyz - k_offset[:,None,None]
    
    # Redefine this as complex so the square root properly gives
    # k>k0 components imaginary frequencies    
    k0 = np.complex128(k0)

    # Finally, generate the propagator!
    propagator = np.exp(1j*np.sqrt(k0**2 - np.linalg.norm(K_xyz,axis=0)**2)
                        * offset_perpendicular)
    propagator *= phase_mask
    
    # Take the conjugate explicitly here instead of negating
    # the previous expression to ensure that complex frequencies
    # get mapped to values <1 instead of >1
    propagator = complex_to_torch(np.conj(propagator)) 
    
    return propagator.to(**kwargs)


def near_field(wavefront, angular_spectrum_propagator):
    """ Propagates a wavefront via the angular spectrum method

    This function accepts an 3D torch tensor, where the last dimension
    represents the real and imaginary components of the wavefield, and
    returns the near-field propagated version of it. It does this
    using the supplied angular spectrum propagator, which is a premade
    phase mask.


    Parameters
    ----------
    wavefront : torch.Tensor
        The JxNxMx2 stack of complex wavefronts to be propagated
    angular_spectrum_propagator : torch.Tensor
        The NxM phase mask to be applied during propagation

    Returns
    -------
    propagated : torch.Tensor
        The propagated wavefront 
    """
    return t.ifft(cmult(angular_spectrum_propagator,t.fft(wavefront,2)), 2)



def inverse_near_field(wavefront, angular_spectrum_propagator):
    """ Inverse propagates a wavefront via the angular spectrum method

    This function accepts an 3D torch tensor, where the last dimension
    represents the real and imaginary components of the wavefield, and
    returns the near-field propagated version of it. It does this
    using the supplied angular spectrum propagator, which is a premade
    phase mask.

    It propagates the wave using the conjugate of the supplied phase mask,
    which corresponds to the inverse propagation problem.


    Parameters
    ----------
    wavefront : torch.Tensor
        The JxNxMx2 stack of complex wavefronts to be propagated
    angular_spectrum_propagator : torch.Tensor
        The NxM phase mask to be applied during propagation

    Returns
    -------
    propagated : torch.Tensor
        The inverse propagated wavefront
    """
    return t.ifft(cmult(t.fft(wavefront,2), cconj(angular_spectrum_propagator)), 2)



# I think it would be worthwhile to implement an FFT-DI based strategy as
# well, especially for probe initialization where the propagation distance
# can be large relative to what the angular spectrum method can reliably handle



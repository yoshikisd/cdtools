"""This module contains various propagators for light fields

All the functions here are designed for use in an automatic differentiation
ptychography model. Each function implements a different propagator.
"""
from __future__ import division, print_function, absolute_import

from CDTools.tools.cmath import *
import torch as t
from torch.nn.functional import grid_sample
from scipy import fftpack
import numpy as np
from matplotlib import pyplot as plt

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


def generate_high_NA_k_intensity_map(sample_basis, det_basis,det_shape,distance, wavelength, *args, **kwargs):
    """Generates k-space and intensity maps to allow for high-NA far-field propagation of light

    At high numerical apertures or for very tilted samples, the simple
    linear map between location on the detector and location in k-space
    starts to break down. In addition, at angles above roughly 15 degrees,
    a correction is needed to account for the decreasing solid angle
    intersected by each pixel on the detector.
    
    This function generates a map which can be used to apply both corrections
    via the high_NA_far_field propagator. The k-map which is output is
    defined as a map between pixel location on the detector and locations
    in the output of the standard, low-NA far-field propagated wavefield.
    The output coordinate system is defined to run from -1 to 1 in both
    directions - this allows for compatibility with pytorch's grid_sample
    function. Some detector pixels may be mapped to values outside the
    rangel [-1,1], depending on the respective sample and detector geometries.
    This is most likely to be the case if the major correction is due to a
    tilted sample.

    The intensity map is simply an object, the shape of the detector, which
    encodes intensity corrections between 0 and 1 per pixel.

    Parameters
    ----------
    sample_basis: array
        The 3x2 sample basis, in real space
    det_basis: array
        The 3x2 detector basis, in real space
    det_shape: array
        The length-2 shape of the detector array, (N,M)
    distance: float
        The sample-to-detector distance
    wavelength: float
        The wavelength of light being propagated


    Returns
    -------
    k_map : torch.Tensor
        An NxMx2 tensor mapping detector pixels to locations in the low-NA propagated wavefield
    intensity_map : torch.Tensor
        An NxM tensor encoding the high-NA intensity correction
    """

    # First we convert things to numpy
    try:
        sample_basis = sample_basis.detach().cpu().numpy()
    except:
        pass

    try:
        det_basis = det_basis.detach().cpu().numpy()
    except:
        pass

    det_shape = np.array(tuple(det_shape))

    try:
        distance = distance.detach().cpu().numpy()[0]
    except:
        pass

    try:
        wavelength = wavelength.detach().cpu().numpy()[0]
    except:
        pass

    # The next order of business is to calculate the k values associated with
    # each pixel.

    i_arr = np.arange(det_shape[0])
    i_arr = i_arr - np.mean(i_arr)
    j_arr = np.arange(det_shape[1])
    j_arr = j_arr - np.mean(j_arr)
    Is, Js = np.meshgrid(i_arr,j_arr,indexing='ij')
    samp_det_vec = np.cross(det_basis[:,0],det_basis[:,1])
    samp_det_vec *= distance / np.linalg.norm(samp_det_vec)

    Rs = np.tensordot(det_basis,np.stack([Is,Js]),axes=1) \
        + samp_det_vec[:,None,None]

    k0 = 2*np.pi/wavelength
    
    Ks = k0 * Rs / np.linalg.norm(Rs, axis=0)

    # This is the cosine of the angle with the detector normal
    intensity_map = np.tensordot(samp_det_vec/(k0*distance),Ks,axes=1)
    intensity_map = t.Tensor(intensity_map).to(*args, **kwargs)


    # This accounts for the implied phase ramp along the exit wave direction
    # In other words, it prevents the diffraction pattern from sliding off the
    # detector when the sample is tilted but represented by an object with
    # uniform phase.
    Ks -= k0 * samp_det_vec[:,None,None] / distance

    # Now we move on to finding the conversion into k-space
    # for the sample grid. It turns out we can do this by multiplying
    # them with the real space basis (dual of the reciprocal space
    # basis is the real space basis). In fact, because we want to return
    # values scaled to the overall size of the k-space window, we don't
    # even need the shape of the sample array

    k_map = np.tensordot(2*sample_basis.transpose()[::-1,:] / (2*np.pi),Ks,axes=1)
    k_map = t.Tensor(np.moveaxis(k_map,0,2)).to(*args, **kwargs)
    
    # Potentially we need a correction to account for the discrete nature
    # of the FFT
    
    # And finally, we need to convert the results to pytorch
    
    return k_map, intensity_map




def high_NA_far_field(wavefront, k_map, intensity_map=None):
    """Performs a far-field propagation step including a correction for high-NA scenarios

    Two major corrections need to be performed when propagating light fields
    into the far field at high numerical aperture or when the sample is
    tilted as compared to the detector. The first correction is a deviation
    from the linear relationship between detector position and spatial
    frequency in the near field. This is accounted for with the k_map
    argument, as generated by the generate_high_NA_k_intensity_map 
    function.

    The second correction is the change in the solid angle which each pixel
    subtends at high NA. This is accounted for with an optional intensity
    map. This is kept optional because some detectors - specifically, those
    for penetrating radiation - may either not need a correction or need
    a different correction due to the volumetric nature of the pixels.
    
    If the k-map map any pixels on the detector to pixels outside of the
    k-space range of the wavefront, these will be set to zero. This is in
    keeping with the typical assumption that the sample is band-limited to
    the Nyquist frequency for the array on which it is sampled.

    Parameters
    ----------
    wavefront : torch.Tensor
        The JxNxMx2 stack of complex wavefronts propagated to the far-field
    k_map : torch.Tensor
        The NxMx2 map accounting for high NA distortion, as generated by generate_high_NA_k_intensity_map
    intensity_map : torch.Tensor
        The optional NxM tensor accounting for the intensity variation across the detector

    
    Returns
    -------
    propagated : torch.Tensor
        The JxNxMx2 exit wavefield


    """
    low_NA_wavefield = far_field(wavefront)
    # I'm going to need to separately interpolate the real and complex parts
    # This can be done

    k_map = k_map[None,:,:,:]
    # Will only work for a 4D wavefile stack.
    def process_wavefield_stack(low_NA_wavefield):
        real_output = grid_sample(low_NA_wavefield[None,:,:,:,0],k_map,mode='bilinear',padding_mode='zeros', align_corners=False)
        imag_output = grid_sample(low_NA_wavefield[None,:,:,:,1],k_map,mode='bilinear',padding_mode='zeros', align_corners=False)

        result = t.stack((real_output[0,:,:,:],imag_output[0,:,:,:]),dim=3)
    
        if intensity_map is not None:
            result = result * intensity_map[None,:,:,None]

        return result

    original_dim = wavefront.dim()
    if original_dim == 3:
        result = process_wavefield_stack(low_NA_wavefield[None,:,:,:])
        return result[0,:,:,:]
    if original_dim == 4:
        result = process_wavefield_stack(low_NA_wavefield)
        return result
    if original_dim == 5:
        result = []
        for i in range(low_NA_wavefield.size()[0]):
            result.append(process_wavefield_stack(low_NA_wavefield[i,:,:,:,:]))
        return t.stack(result)
    else:
        raise IndexError('Wavefield had incorrect number of dimensions')





def generate_angular_spectrum_propagator(shape, spacing, wavelength, z, *args, remove_z_phase=False, **kwargs):
    """Generates an angular-spectrum based near-field propagator from experimental quantities

    This function generates an angular-spectrum based near field
    propagator that will work on torch Tensors. The function is structured
    this way - to generate the propagator first - because the
    generation of the propagation mask is a bit expensive and if this
    propagator is used in a reconstruction program, then it will be best
    to calculate this mask once and reuse it.

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
    remove_z_phase : bool
        Default False, whether to remove the dominant z-direction phase dependence

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

    if remove_z_phase:
        propagator *= np.exp(-1j * k0 * z)

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

    In addition, if propagate_along_offset is True, there is an assumed phase
    ramp applied to the wavefield before propagation, defined such that a
    feature with uniform phase will propagate along the direction of the 
    defined offset vector. This will also remove the phase variation along
    the propagation direction, because it makes the most physical sense to
    regard this choice as removing the dominant phase variation in 3D, allowing
    for the generation of a smoothly varying wavefield over 3D volumes.
    This decision provides the best numerical stability and allows for the
    simple setup of light fields copropagating with the coordinate system.


    Parameters
    ----------
    shape : array
        The shape of the arrays to be propagated
    basis : array
        The (2x3) set of basis vectors describing the array to be propagated
    wavelength : float
        The wavelength of light to simulate propagation of
    offset_vector : array
        The displacement to propagate the wavefield along.
    propagate_along_offset : bool
        Optional, whether to include an implied phase ramp to propagate uniform phase features along the offset direction

    Returns
    -------
    propagator : torch.Tensor
        A phase mask which accounts for the phase change that each plane wave will undergo.
    """

    # These check for any pytorch inputs and convert them
    try:
        basis = basis.detach().cpu().numpy()
    except:
        pass

    shape = np.array(tuple(shape))

    try:
        offset_vector = offset_vector.detach().cpu().numpy()
    except:
        pass

    try:
        wavelength = wavelength.detach().cpu().numpy()[0]
    except:
        pass


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

    if np.linalg.norm(offset_vector) == 0:
        # for numerical stability if this is exactly zero we need a special case
        k_offset = np.array([0,0,0]) 
    else:  
        k_offset = offset_parallel * k0 / np.linalg.norm(offset_vector)


    # There apparently is a sign correction that I need to apply
    sign_correction = np.sign(np.dot(perpendicular_dir,offset_vector))
    
    # Only implement the shift if the flag is set to True
    if propagate_along_offset:
        K_xyz = K_xyz + k_offset[:,None,None] * sign_correction

        # we also need to remove the z-dependence on the phase
        # This time, though, the z-dependence actually has to do with
        # the oput of plane component of k at the central offset. Normally
        # this is 0, so the z-component is just k0, but not in this case
        phase_mask *= np.exp(-1j * np.sqrt(k0**2 - np.linalg.norm(k_offset)**2)
                             * offset_perpendicular)
    
    # Redefine this as complex so the square root properly gives
    # k>k0 components imaginary frequencies    
    k0 = np.complex128(k0)

    # Finally, generate the propagator!
    propagator = np.exp(1j*np.sqrt(k0**2 - np.linalg.norm(K_xyz,axis=0)**2)
                        * offset_perpendicular)
    propagator *= phase_mask
    
    # This removes the z-dependence on the phase:
    #if propagate_along_offset:
    #    print(offset_perpendicular)
    #    propagator *= np.exp(-1j * k0 * offset_perpendicular)
    
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



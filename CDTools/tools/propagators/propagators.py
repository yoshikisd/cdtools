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
           'inverse_far_field', 'inverse_near_field',
           'generate_high_NA_k_intensity_map',
           'high_NA_far_field',
           'generate_generalized_angular_spectrum_propagator']


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


def generate_high_NA_k_intensity_map(sample_basis, det_basis,det_shape,distance, wavelength, *args, lens=False, **kwargs):
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

    If the optional "lens" parameter is set to True, the intensity map will
    be set to a uniform map, and the distortion of Fourier space due to the
    flat nature of the detector (that is, the portion of the distortion
    that exists even if the sample is not tilted) will be disabled. This is
    to account for the fact that a good, infinity-conjugate imaging lens
    will do it's best to correct for these abberations in the lens. Of course,
    the lens will not be perfect, but in such a case it is a better
    approximation to assume that the lens is perfect than to assume that it
    is not there at all.

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
    lens: bool
        Whether the diffraction pattern is formed by a lens or not.

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

    # This could potentially correct for a mistake in the implied
    # propagation direction (e.g. choosing e^ikx instead of e^-ikx)
    # This appears to be correct, based on empirical evidence from
    # a grazing incidence reflection experiment at 10 degrees
    # on the optical table
    samp_det_vec *= -1

    if lens == False:
        # This correctly reproduces the sample-to-each-pixel vectors
        # in the case where the diffraction pattern is actually formed
        # by Fraunhoffer diffraction
        Rs = np.tensordot(det_basis,np.stack([Is,Js]),axes=1) \
            + samp_det_vec[:,None,None]
    else:
        # This forms a distorted set of vectors designed to produce the
        # correct Fourier space map in the case where an imaging lens is
        # used in the 2f geometry. One should not read too much meaning
        # into these vectors, they are simply set up to produce the
        # correct final K-map
        Rs = np.tensordot(det_basis,np.stack([Is,Js]),axes=1)#
        Rs += (samp_det_vec / np.linalg.norm(samp_det_vec))[:,None,None] * \
            np.sqrt(np.sum((samp_det_vec)**2)-np.sum(Rs**2,axis=0))[None,:,:]
        
    k0 = 2*np.pi/wavelength
    
    Ks = k0 * Rs / np.linalg.norm(Rs, axis=0)

    # My attempt at seeing what happens if I flip the Ks
    #Ks *= -1
    
    # This is the cosine of the angle with the detector normal
    intensity_map = np.tensordot(samp_det_vec/(k0*distance),Ks,axes=1)
    if lens:
        # Set the intensity map to be uniform if a lens is being used
        intensity_map = np.ones_like(intensity_map)
    
    intensity_map = t.Tensor(intensity_map).to(*args, **kwargs)

    
    # This accounts for the implied phase ramp along the exit wave direction
    # In other words, it prevents the diffraction pattern from sliding off the
    # detector when the sample is tilted but represented by an object with
    # uniform phase.
    Ks -= k0 * samp_det_vec[:,None,None] / distance

    # A potential alternative when Ks are flipped
    #Ks += k0 * samp_det_vec[:,None,None] / distance

    
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
    
    If the k-map maps any pixels on the detector to pixels outside of the
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
    #plt.figure()
    #plt.pcolormesh(k_map[0,:,:,0].cpu().numpy(),k_map[0,:,:,1].cpu().numpy(),
    #               np.ones_like(k_map[0,:-1,:-1,0].cpu().numpy()))
    #plt.show()
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





def generate_angular_spectrum_propagator(shape, spacing, wavelength, z, *args, remove_z_phase=False, bandlimit=None, **kwargs):
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
    
    If the optional bandlimit parameter is set, the propagator will be set
    to zero beyond an explicit bandlimiting frequency. This is helpful if the
    propagator will be used in a repeated multiply/propagate framework such
    as a multislice algorithm, where it helps to prevent aliasing.

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
    bandlimit : float
        Optional, a fraction of the full detector radius beyond which to set the propagator to zero.

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

    # Properly accuount for evanescent waves
    if z >=0:
        propagator = np.exp(1j*np.sqrt(k0**2 - Ki**2 - Kj**2) * z)
    else:
        propagator = np.exp(1j*np.conj(np.sqrt(k0**2 - Ki**2 - Kj**2)) * z)
        
    if remove_z_phase:
        propagator *= np.exp(-1j * k0 * z)

    if bandlimit is not None:
        Rs = np.sqrt((Ki / np.max(ki))**2 + (Kj / np.max(kj))**2)
        propagator = propagator * (Rs < bandlimit)
        
    # Take the conjugate explicitly here instead of negating
    # the previous expression to ensure that complex frequencies
    # get mapped to values <1 instead of >1
    propagator = complex_to_torch(np.conj(propagator)) 
    
    return propagator.to(*args, **kwargs)


def generate_generalized_angular_spectrum_propagator(shape, basis, wavelength, offset_vector, *args, propagation_vector=None, propagate_along_offset=False, **kwargs):
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


    In addition, if the propagation_vector is set, there is an assumed
    phase ramp  applied to the wavefield before propagation, defined such
    that a feature with uniform phase will propagate along the direction of
    the propagation vector. This will also remove the phase variation along
    the propagation direction, because it makes the most physical sense to
    regard this choice as removing the dominant phase variation in 3D, allowing
    for the generation of a smoothly varying wavefield over 3D volumes.
    This decision provides the best numerical stability and allows for the
    simple setup of light fields copropagating with the coordinate system.
    
    If the propagate_along_offset option is set to True, then the propagation
    vector will be set equal to the offset vector. This overrides the
    propagation_vector option

    Note that, unlike in the case of the simple angular spectrum propagator,
    the direction of "forward propagation" is defined by the offset vector.
    Therefore, in the simple case of a perpendicular offset, there will be
    no difference between using an offset vector or the negative of the 
    offset vector. This is because, for the light propagation problem to
    be well posed, the assumption must be made that light only passes through
    the plane of the known wavefield in one direction. Mathematically, this
    corresponds to a choice of uniform phase objects either accumulating
    positive or negative phase. In the simple propagation case, there is
    no ambiguity introduced by always choosing the light field to propagate
    along the positive z direction. In the general case, there is no equivalent
    obvious choice - thus, the light is always assumed to pass through the
    initial plane travelling in the direction of the final plane.

    Practically, if one wants to simulate inverse propagation, there are then
    two possible approaches. First, one can use the inverse_near_field 
    function, which simulates the inverse propagation problem and therefore
    will naturally simulate propagation in the opposite direction. Second,
    one can explicitly include a propagation_vector argument, which overrides
    the offset vector in defining the direction in which light passes through
    the input plane. However, in this case, the resulting light field will have
    the overall phase accumulation due to propagation along the propagation
    vector removed, which may not be the intended behavior. However, this is
    not recommended, as inverse propagation will tend to magnify evanescent
    waves - it is therefore preferable (unless there is a specific need to
    account for evanescent waves properly) to use the inverse near field
    propagator
    

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
    propagation_vector : array
        The vector along which to include an implied phase ramp to propagate uniform phase features along, if set
    propagate_along_offset : bool
        Overrides propagation_vector, sets the propagation vector to equal the offset vector if set.

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

    if propagation_vector is not None:
        try:
            propagation_vector = propagation_vector.detach().cpu().numpy()
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
    phase_mask = np.exp(-1j * np.tensordot(offset_vector,K_xyz,axes=1))
    
    # Next, we apply a shift to the k-space vectors which sets up
    # propagation such that a uniform phase object will propagate along the
    # propagation axis. This is not modeling a physical effect, but simply is
    # the clearest way to do a rigorous simulation while preventing
    # aliasing-related challenges. If used (as is by default), be aware
    # and prepare the input wavefields appropriately.
    if propagate_along_offset:
        propagation_vector = offset_vector
        
    perpendicular_dir = np.cross(basis[:,1],basis[:,0])
    perpendicular_dir /= np.linalg.norm(perpendicular_dir)
    offset_perpendicular = np.dot(perpendicular_dir, offset_vector)


    k0 = 2*np.pi/wavelength

    sign_correction = 1

    # Only implement the shift if the flag is set to True
    if propagation_vector is not None:
            
        prop_perpendicular = np.dot(perpendicular_dir, propagation_vector)
        prop_parallel = propagation_vector - perpendicular_dir \
            * prop_perpendicular
        
        
        if np.linalg.norm(propagation_vector) == 0:
            # for numerical stability if this is exactly zero we need
            # a special case
            k_offset = np.array([0,0,0]) 
        else:  
            k_offset = prop_parallel * k0 / np.linalg.norm(propagation_vector)

        K_xyz = K_xyz - k_offset[:,None,None] 

        # There apparently is a sign correction that I need to apply
        #sign_correction = np.sign(np.dot(perpendicular_dir,propagation_vector))
        sign_correction = np.sign(np.dot(offset_vector,propagation_vector))

        # we also need to remove the z-dependence on the phase
        # This time, though, the z-dependence actually has to do with
        # the out of plane component of k at the central offset. Normally
        # this is 0, so the z-component is just k0, but not in this case
        # We only need one case here, unlike with the propagator, because
        # k_offset will always be less than k0
        phase_mask *= np.exp(-1j * np.sqrt(k0**2 - np.linalg.norm(k_offset)**2)
                             * sign_correction
                             * np.abs(offset_perpendicular))
            
    
    # Redefine this as complex so the square root properly gives
    # k>k0 components imaginary frequencies    
    k0 = np.complex128(k0)
    
    # Finally, generate the propagator!
    # Must have cases to ensure that evanescent waves decay instead of grow
    if sign_correction > 0:
        propagator = np.exp(1j*np.sqrt(k0**2 - np.linalg.norm(K_xyz,axis=0)**2)
                            * sign_correction * np.abs(offset_perpendicular))
    else:
        propagator = np.exp(-1j * np.conj(np.sqrt(k0**2 -
                                        np.linalg.norm(K_xyz,axis=0)**2))
                            * np.abs(offset_perpendicular))
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

    It propagates the wave using the complex conjugate of the supplied
    phase mask. This corresponds to propagation backward across the original
    propagation region - however, the treatment of evanescent waves is such
    that evanescent waves will decay both during the forward propagation and
    inverse propagation. This is done for reasons of numerical stability,
    as the choice to magnify evanescent waves during the inverse propagation
    process will quickly lead to magnification of any small amount of noise at 
    frequencies larger than k_0, and in most typical situations will even
    lead to overflow of the floating point range. If evanescent waves need
    to be treated appropriately for any reason, it is recommended to use the
    "magnify_evanescent" option in the appropriate helper function used to
    generate the propagation phase mask. In this case, evanescent waves will
    be magnified both when used with the forward and inverse near field
    functions


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


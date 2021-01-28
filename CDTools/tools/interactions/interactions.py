""" This module contains various to simulate stages in the probe-sample interaction

All the tools here are designed to work with automatic differentiation. Each
function simulates some aspect of an interaction model that can be used
for ptychographic reconstruction.
"""

from __future__ import division, print_function, absolute_import

from CDTools.tools.cmath import *
import torch as t
import numpy as np
from CDTools.tools import propagators 

__all__ = ['translations_to_pixel', 'pixel_to_translations',
           'project_translations_to_sample',
           'ptycho_2D_round','ptycho_2D_linear','ptycho_2D_sinc',
           'RPI_interaction']



def translations_to_pixel(basis, translations, surface_normal=t.Tensor([0.,0.,1.])):
    """Takes real space translations and outputs them in pixel space
    
    This works for any 2D ptychography geometry. It takes in
    A set of translations in (x,y) space and outputs the same translations
    in internal pixel units perpendicular to the detector. 
    
    It uses information on the wavefield basis and, if defined, the
    sample normal, to perform the conversion.
    
    The assumed geometry is incoming radiation with a wavevector parallel
    to the +z axis, [0,0,1]. The default sample orientation has a surface
    normal parallel to this direction

    Parameters
    ----------
    basis : torch.Tensor
        The real space basis the wavefields are defined in
    translations : torch.Tensor 
        A Jx3 stack of real-space translations, or a single translation
    surface_normal : torch.Tensor
        Optional, the sample's surface normal

    Returns
    -------
    pixel_translations : torch.Tensor
        A Jx2 stack of translations in internal (i,j) pixel-space, or a single translation
    """
    
    projection_1 = t.Tensor([[1,0,0],
                             [0,1,0],
                             [0,0,0]]).to(device=translations.device,dtype=translations.dtype)
    projection_2 = t.inverse(t.Tensor([[1,0,0],
                                       [0,1,0],
                                       -surface_normal/
                                       surface_normal[2]])).to(device=translations.device,dtype=translations.dtype)
    basis_vectors_inv = t.pinverse(basis).to(device=translations.device,
                                             dtype=translations.dtype)
    projection = t.mm(basis_vectors_inv,
                      t.mm(projection_2,projection_1))
    projection = projection.t()

    single_translation = False
    if len(translations.shape) == 1:
        translations = translations[None,:]
        single_translation = True

    pixel_translations = t.mm(translations, projection)

    if single_translation:
        return pixel_translations[0]
    else:
        return pixel_translations
    

def pixel_to_translations(basis, pixel_translations, surface_normal=t.Tensor([0,0,1])):
    """Takes pixel-space translations and outputs them in real space
    
    This works for any 2D ptychography geometry. It takes in
    A set of internal pixel unit translations in (i,j) space and
    outputs the same translations real (x,y) space
    
    It uses information on the wavefield basis and, if defined, the
    sample normal, to perform the conversion.
    
    The assumed geometry is incoming radiation with a wavevector parallel
    to the +z axis, [0,0,1]. The default sample orientation has a surface
    normal parallel to this direction. Because of this, the z direction
    translation is always set to zero in the conversion

    Parameters
    ----------
    basis : torch.Tensor
        The real space basis the wavefields are defined in
    translations : torch.Tensor
        A Jx2 stack of pixel-space translations, or a single translation
    surface_normal : torch.Tensor
        Optional, the sample's surface normal

    Returns
    -------
    real_translations : torch.Tensor 
        A Jx3 stack of real-space translations, or a single translation
    """
    projection_1 = t.Tensor([[1,0,0],
                             [0,1,0],
                             [0,0,0]]).to(device=basis.device,dtype=basis.dtype)
    projection_2 = t.inverse(t.Tensor([[1,0,0],
                                       [0,1,0],
                                       -surface_normal/
                                       surface_normal[2]])).to(device=basis.device,dtype=basis.dtype)
    basis_vectors_inv = t.pinverse(basis)
    projection = t.mm(basis_vectors_inv,
                      t.mm(projection_2,projection_1))
    # Literally just need the pseudoinverse of the projection we used to go
    # the other way
    projection = t.pinverse(projection).t()

    single_translation = False
    if len(pixel_translations.shape) == 1:
        pixel_translations = pixel_translations[None,:]
        single_translation = True

    translations = t.mm(pixel_translations, projection)

    if single_translation:
        return translations[0]
    else:
        return translations



def project_translations_to_sample(sample_basis, translations):
    """Takes real space translations and outputs them in pixels in a sample basis
    
    This projection function is designed for the Bragg2DPtycho class. More
    broadly, it works to take a set of translations in the lab frame and
    convert each one into two values. First, an (i,j) value in pixels
    describing the location of the probe's intersection with the sample
    plane, assuming the basis found in sample_basis is used. Second, an
    amount that the probe must be propagated along the z-axis to reach
    the sample plane at the given location. This includes both the effect of
    the tilted sample plane and any explicitly defined motion along the z axis
    of the probe-forming optic as included in the input translations.


    Note that because of the sign convention (that this function returns th
    relative amount the probe needs to be propagated to reach any given
    location), a positive motion along the z-axis of the probe forming optics
    will lead to a negative propagation distance.
    
    The assumed geometry is incoming radiation with a wavevector parallel
    to the +z axis, [0,0,1].

    Parameters
    ----------
    sample_basis : torch.Tensor
        The real space basis the wavefields are defined in
    translations : torch.Tensor 
        A Jx3 stack of real-space translations, or a single translation

    Returns
    -------
    pixel_translations : torch.Tensor
        A Jx2 stack of translations in internal (i,j) pixel-space, or a single translation
    propagations : torch.Tensor
        A length-J vector of propagation distances, in meters
    """

    # Must do this all in pytorch unfortunately
    # First we calculate the surface normal for the projection onto the sample
    surface_normal =  t.cross(sample_basis[:,1],sample_basis[:,0])
    surface_normal /= t.norm(surface_normal)


    # Then we calculate a matrix which can do the projection
    
    propagation_dir = t.Tensor(np.array([0,0,1])).to(
        device=surface_normal.device,
        dtype=surface_normal.dtype)

    I = t.eye(3).to(
        device=surface_normal.device,
        dtype=surface_normal.dtype)
    
    # Here we're setting up a matrix-vector equation mat*answer=input
    # At some point ger will need to be replaced by outer, but for now
    # outer many places still don't have new enough versions of torch.
    mat = t.cat((I - t.ger(propagation_dir,propagation_dir),
                 surface_normal.unsqueeze(0)))
    
    # And we invert the matrix to do the projection
    projector = t.pinverse(mat)[:,:3].to(device=translations.device,
                                         dtype=translations.dtype)

    # Finally, we need to get the result in the right basis, so this will
    # do that conversion for us
    basis_vectors_inv = t.pinverse(sample_basis).to(device=translations.device,
                                                    dtype=translations.dtype)

    # Same for the propagation direction to extract how far it must propagate
    propagation_dir_inv = propagation_dir.unsqueeze(0).to(
        device=translations.device,
        dtype=translations.dtype)

    
    sample_projection = t.mm(basis_vectors_inv, projector).t()
    prop_projection = t.mm(propagation_dir_inv, projector).t()


    single_translation = False
    if len(translations.shape) == 1:
        translations = translations[None,:]
        single_translation = True

    pixel_translations = t.mm(translations, sample_projection)
    propagations = t.mm(translations, prop_projection) \
        - t.mm(translations,propagation_dir[:,None])

    if single_translation:
        return pixel_translations[0], propagations[0]
    else:
        return pixel_translations, propagations
    

    

def ptycho_2D_round(probe, obj, translations):
    """Returns a stack of exit waves without accounting for subpixel shifts

    This function returns a collection of exit waves, with the first
    dimension as the translation index and the final dimensions
    corresponding to the detector. The exit waves are calculated by
    shifting the probe by the rounded value of the translation
    
    Parameters
    ----------
    probe : torch.Tensor
        A (P)xMxL probe function for the exit waves
    object : torch.Tensor
        The object function to be probed
    translations : torch.Tensor
        The (N)x2 array of (i,j) translations to simulate
    
    Returns
    -------
    exit_waves : torch.Tensor 
        An (N)x(P)xMxL tensor of the calculated exit waves
    """

    single_translation = False
    if translations.dim() == 1:
        translations = translations[None,:]
        single_translation = True

        
    integer_translations = t.round(translations).to(dtype=t.int32)
    selections = t.stack([obj[tr[0]:tr[0]+probe.shape[-3],
                              tr[1]:tr[1]+probe.shape[-2]]
                          for tr in integer_translations])

    if single_translation:
        return cmult(probe,selection)[0]
    else:
        return cmult(probe,selections)




def ptycho_2D_linear(probe, obj, translations, shift_probe=True):
    """Returns a stack of exit waves accounting for subpixel shifts
 
    This function returns a collection of exit waves, with the first
    dimension as the translation index and the final dimensions
    corresponding to the detector. The exit waves are calculated by
    shifting the probe with each translation in turn, using linear
    interpolation to combine the results

    If shift_probe is True, it applies the subpixel shift to the probe,
    otherwise the subpixel shift is applied to the object

    Parameters
    ----------
    probe : torch.Tensor
        An MxL probe function for the exit waves
    object : torch.Tensor
        The object function to be probed
    translations : torch.Tensor
        The Nx2 array of translations to simulate
    shift_probe : bool
        Default True, Whether to subpixel shift the probe or object

    Returns
    -------
    exit_waves : torch.Tensor
        An NxMxL tensor of the calculated exit waves
    """
    single_translation = False
    if translations.dim() == 1:
        translations = translations[None,:]
        single_translation = True
        
    # Separate the translations into a part that chooses the window
    # And a part that defines the windowing function
    integer_translations = t.floor(translations)
    subpixel_translations = translations - integer_translations
    integer_translations = integer_translations.to(dtype=t.int32)

    exit_waves = []
    if shift_probe:
        for tr, sp in zip(integer_translations,
                          subpixel_translations):
            # This isn't perfectly symmetric but I think it's okay for now
            # It should get the job done
            # Basically, we shift the probe's position by a subpixel (i,j),
            # rolling the edges of the array, and use that to multiply
            # by the object
            sel00 = probe[:,:]
            sel01 = t.cat((probe[:,-1:],probe[:,:-1]),dim=1)
            sel10 = t.cat((probe[-1:,:],probe[:-1,:]),dim=0)
            sel11 = t.cat((sel01[-1:,:],sel01[:-1,:]),dim=0)
            
            selection = sel00 * (1-sp[0])*(1-sp[1]) + \
                sel10 * sp[0]*(1-sp[1]) + \
                sel01 * (1-sp[0])*sp[1] + \
                sel11 * sp[0]*sp[1]
            
            obj_slice = obj[tr[0]:tr[0]+probe.shape[0],
                            tr[1]:tr[1]+probe.shape[1]]
            
            exit_waves.append(cmult(selection,obj_slice))
    else:
        for tr, sp in zip(integer_translations,
                          subpixel_translations):
            #
            # Here we subpixel shift the object by (-i,-j) after
            # slicing out the correct translation of the probe
            #
            
            sel00 = obj[tr[0]:tr[0]+probe.shape[0],
                        tr[1]:tr[1]+probe.shape[1]]
            
            sel01 = obj[tr[0]:tr[0]+probe.shape[0],
                        tr[1]+1:tr[1]+1+probe.shape[1]]
            
            sel10 = obj[tr[0]+1:tr[0]+1+probe.shape[0],
                        tr[1]:tr[1]+probe.shape[1]]
            
            sel11 = obj[tr[0]+1:tr[0]+1+probe.shape[0],
                        tr[1]+1:tr[1]+1+probe.shape[1]]

            selection = sel00 * (1-sp[0])*(1-sp[1]) + \
                sel01 * (1-sp[0])*sp[1] + \
                sel10 * sp[0]*(1-sp[1]) + \
                sel11 * sp[0]*sp[1]

            exit_waves.append(cmult(probe,selection))

    if single_translation:
        return exit_waves[0]
    else:
        return t.stack(exit_waves)




def ptycho_2D_sinc(probe, obj, translations, shift_probe=True, padding=10):
    """Returns a stack of exit waves accounting for subpixel shifts
 
    This function returns a collection of exit waves, with the first
    dimension as the translation index and the final dimensions
    corresponding to the detector. The exit waves are calculated by
    shifting the probe with each translation in turn, using sinc
    interpolation (done via multiplication with a complex exponential
    in Fourier space)

    If shift_probe is True, it applies the subpixel shift to the probe,
    otherwise the subpixel shift is applied to the object

    Parameters
    ----------
    probe : torch.Tensor
        An MxL probe function for the exit waves
    object : torch.Tensor
        The object function to be probed
    translations : torch.Tensor
        The Nx2 array of translations to simulate
    shift_probe : bool
        Default True, Whether to subpixel shift the probe or object
    padding : int
         Default 10, if shifting the object, the padding to apply to the object to avoid circular shift effects

    Returns
    -------
    exit_waves : torch.Tensor
        An NxMxL tensor of the calculated exit waves
    """
    single_translation = False
    if translations.dim() == 1:
        translations = translations[None,:]
        single_translation = True
        
    # Separate the translations into a part that chooses the window
    # And a part that defines the windowing function
    integer_translations = t.floor(translations)
    subpixel_translations = translations - integer_translations
    integer_translations = integer_translations.to(dtype=t.int32)
    
    exit_waves = []
    if shift_probe:
        i = t.arange(probe.shape[0]) - probe.shape[0]//2
        j = t.arange(probe.shape[1]) - probe.shape[1]//2
        I,J = t.meshgrid(i,j)
        I = 2 * np.pi * I.to(t.float32) / probe.shape[0]
        J = 2 * np.pi * J.to(t.float32) / probe.shape[1]
        I = I.to(dtype=probe.dtype,device=probe.device)
        J = J.to(dtype=probe.dtype,device=probe.device)
        
        for tr, sp in zip(integer_translations,
                          subpixel_translations):
            fft_probe = fftshift(t.fft(probe, 2))
            shifted_fft_probe = cmult(fft_probe, expi(-sp[0]*I - sp[1]*J))
            shifted_probe = t.ifft(ifftshift(shifted_fft_probe),2)

            obj_slice = obj[tr[0]:tr[0]+probe.shape[0],
                            tr[1]:tr[1]+probe.shape[1]]

            exit_waves.append(cmult(shifted_probe, obj_slice))
        
    else:
        raise NotImplementedError('Object shift not yet implemented')

    if single_translation:
        return exit_waves[0]
    else:
        return t.stack(exit_waves)


def ptycho_2D_sinc_s_matrix(probe, s_matrix, translations, shift_probe=True, padding=10):
    """Returns a stack of exit waves accounting for subpixel shifts
 
    This function returns a collection of exit waves, with the first
    dimension as the translation index and the final dimensions
    corresponding to the detector. The exit waves are calculated by
    shifting the probe with each translation in turn, using sinc
    interpolation (done via multiplication with a complex exponential
    in Fourier space)

    If shift_probe is True, it applies the subpixel shift to the probe,
    otherwise the subpixel shift is applied to the object

    This needs to be edited to use a slightly different meaning of the s-wave
    format. Currently, each pixel in the latter two dimensions index a location
    on the input wavefield, and the first two indexes index differences from
    that pixel. It is easier to interpret the resulting matrix though if the
    latter two indices index locations in the output plane. NOTE: I believe
    this change has now been made 

    Parameters
    ----------
    probe : torch.Tensor
        An MxL probe function for the exit waves
    s_matrix : torch.Tensor
        The 4D S-Matrix tensor (2B+1x2B+1xObject Shape) to be probed
    translations : torch.Tensor
        The Nx2 array of translations to simulate
    shift_probe : bool
        Default True, Whether to subpixel shift the probe or object
    padding : int
         Default 10, if shifting the object, the padding to apply to the object to avoid circular shift effects

    Returns
    -------
    exit_waves : torch.Tensor
        An NxMxL tensor of the calculated exit waves
    """
    single_translation = False
    if translations.dim() == 1:
        translations = translations[None,:]
        single_translation = True
        
    # Separate the translations into a part that chooses the window
    # And a part that defines the windowing function
    integer_translations = t.floor(translations)
    subpixel_translations = translations - integer_translations
    integer_translations = integer_translations.to(dtype=t.int32)
    
    exit_waves = []

    B = s_matrix.shape[0]//2
    
    if shift_probe:
        i = t.arange(probe.shape[0]) - probe.shape[0]//2
        j = t.arange(probe.shape[1]) - probe.shape[1]//2
        I,J = t.meshgrid(i,j)
        I = 2 * np.pi * I.to(t.float32) / probe.shape[0]
        J = 2 * np.pi * J.to(t.float32) / probe.shape[1]
        I = I.to(dtype=probe.dtype,device=probe.device)
        J = J.to(dtype=probe.dtype,device=probe.device)
        
        for tr, sp in zip(integer_translations,
                          subpixel_translations):
            fft_probe = fftshift(t.fft(probe, 2))
            shifted_fft_probe = cmult(fft_probe, expi(-sp[0]*I - sp[1]*J))
            shifted_probe = t.ifft(ifftshift(shifted_fft_probe),2)
            
            s_matrix_slice = s_matrix[:,:,tr[0]:tr[0]+probe.shape[0]+2*B,
                                      tr[1]:tr[1]+probe.shape[1]+2*B]


            output = t.zeros([probe.shape[0]+2*B,probe.shape[1]+2*B,2]).to(
                                  device=s_matrix_slice.device,
                                  dtype=s_matrix_slice.dtype)

            
            for i in range(s_matrix.shape[0]):
                for j in range(s_matrix.shape[1]):
                    output [i:i+probe.shape[0],j:j+probe.shape[1]] += \
                        cmult(shifted_probe, s_matrix_slice[i,j,i:i+probe.shape[0],j:j+probe.shape[1],:])
            
            #output = t.zeros([s_matrix_slice.shape[2]+2*B,
            #                  s_matrix_slice.shape[3]+2*B,2]).to(
            #                      device=s_matrix_slice.device,
            #                      dtype=s_matrix_slice.dtype)
            
            #for i in range(s_matrix.shape[0]):
            #    for j in range(s_matrix.shape[1]):
            #        output[i:i+probe.shape[0],j:j+probe.shape[1]] += \
            #            cmult(shifted_probe, s_matrix_slice[i,j,:,:,:])

            exit_waves.append(output)
            #exit_waves.append(cmult(shifted_probe, obj_slice))
        
    else:
        raise NotImplementedError('Object shift not yet implemented')

    if single_translation:
        return exit_waves[0]
    else:
        return t.stack(exit_waves)
    

def RPI_interaction(probe, obj):
    """Returns an exit wave from a high-res probe and a low-res obj
 
    In this interaction, the probe and object arrays are assumed to cover
    the same physical region of space, but with the probe array sampling that
    region of space more finely. Thus, to do the interaction, the object
    is first upsampled by padding it in Fourier space (equivalent to a sinc
    interpolation) before being multiplied with the probe. This function is
    called RPI_interaction because this interaction is central to the RPI
    method and is not commonly used elsewhere.

    This also works with object functions that have an extra first dimension
    for an incoherently mixing model. 


    Parameters
    ----------
    probe : torch.Tensor
        An MxL probe function for simulating the exit waves
    obj : torch.Tensor
        An M'xL' or NxM'xL' object function for simulating the exit waves

    Returns
    -------
    exit_wave : torch.Tensor
        An MxL tensor of the calculated exit waves
    """

    # TODO: The upsampling only works for arrays of even dimension!
    
    # The far-field propagator is just a 2D FFT but with an fftshift
    fftobj = propagators.far_field(obj)
    # We calculate the padding that we need to do the upsampling
    pad0l = (probe.shape[-3] - obj.shape[-3])//2
    pad0r = probe.shape[-3] - obj.shape[-3] - pad0l
    pad1l = (probe.shape[-2] - obj.shape[-2])//2
    pad1r = probe.shape[-2] - obj.shape[-2] - pad1l
        
    if obj.dim() == 3:
        fftobj = t.nn.functional.pad(fftobj, (0, 0, pad1l, pad1r, pad0l, pad0r))
    elif obj.dim() == 4:
        fftobj = t.nn.functional.pad(
            fftobj, (0, 0, pad1l, pad1r, pad0l, pad0r, 0, 0))
    else:
        raise NotImplementedError('RPI interaction with obj of dimension higher than 4 (including complex dimension) is not supported.')
        
    # Again, just an inverse FFT but with an fftshift
    upsampled_obj = propagators.inverse_far_field(fftobj)

    if obj.dim() == 4:
        return cmult(probe[None,...], upsampled_obj)
    else:
        return cmult(probe, upsampled_obj)

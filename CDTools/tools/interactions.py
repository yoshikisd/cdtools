from __future__ import division, print_function, absolute_import

from CDTools.tools.cmath import *
import torch as t
import numpy as np


__all__ = ['translations_to_pixel', 'pixel_to_translations',
           'ptycho_2D_round','ptycho_2D_linear','ptycho_2D_sinc']

#
# This file will host tools to turn various kinds of model information
# (probe, 2D object, 3D object, etc) into exit waves leaving the sample
# area.
#


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

    Args:
        basis (torch.Tensor) : The real space basis the wavefields are defined in
        translations (torch.Tensor) : A Jx3 stack of real-space translations
        surface_normal (torch.Tensor) : Optional, the sample's surface normal
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

    Args:
        basis (torch.Tensor) : The real space basis the wavefields are defined in
        translations (torch.Tensor) : A Jx2 stack of pixel-space translations
        surface_normal (torch.Tensor) : Optional, the sample's surface normal
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

    
def ptycho_2D_round(probe, obj, translations):
    """Returns a stack of exit waves without accounting for subpixel shifts

    This function returns a collection of exit waves, with the first
    dimension as the translation index and the final dimensions
    corresponding to the detector. The exit waves are calculated by
    shifting the probe by the rounded value of the translation
    
    Args:
        probe (torch.Tensor) : An MxL probe function for the exit waves
        object (torch.Tensor) : The object function to be probed
        translations (torch.Tensor) : The Nx2 array of (i,j) translations to simulate
    
    Returns:
        torch.Tensor : An NxMxL tensor of the calculated exit waves
    """
    single_translation = False
    if translations.dim() == 1:
        translations = translations[None,:]
        single_translation = True
        
    integer_translations = t.round(translations).to(dtype=t.int32)
    selections = [obj[tr[0]:tr[0]+probe.shape[0],
                      tr[1]:tr[1]+probe.shape[1]]
                  for tr in integer_translations]
    if single_translation:
        return [cmult(probe,selection) for selection in selections][0]
    else:
        return t.stack([cmult(probe,selection) for selection in selections])




def ptycho_2D_linear(probe, obj, translations, shift_probe=True):
    """Returns a stack of exit waves accounting for subpixel shifts
 
    This function returns a collection of exit waves, with the first
    dimension as the translation index and the final dimensions
    corresponding to the detector. The exit waves are calculated by
    shifting the probe with each translation in turn, using linear
    interpolation to combine the results

    If shift_probe is True, it applies the subpixel shift to the probe,
    otherwise the subpixel shift is applied to the object

    Args:
        probe (torch.Tensor) : An MxL probe function for the exit waves
        object (torch.Tensor) : The object function to be probed
        translations (torch.Tensor) : The Nx2 array of translations to simulate
        shift_probe (bool) : Whether to subpixel shift the probe or object
    Returns:
        torch.Tensor : An NxMxL tensor of the calculated exit waves
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

    Args:
        probe (torch.Tensor) : An MxL probe function for the exit waves
        object (torch.Tensor) : The object function to be probed
        translations (torch.Tensor) : The Nx2 array of translations to simulate
        shift_probe (bool) : Whether to subpixel shift the probe or object
        padding (int) : Default 10, if shifting the object, the padding to apply to the object to avoid circular shift effects
    Returns:
        torch.Tensor : An NxMxL tensor of the calculated exit waves
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

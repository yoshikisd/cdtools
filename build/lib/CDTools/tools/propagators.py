from __future__ import division, print_function, absolute_import
from CDTools.tools.cmath import *
import torch as t

__all__ = ['far_field', 'near_field', 'inverse_far_field', 'inverse_near_field', 'get_exit_waves']


def far_field(wavefront, detector_shape, detector_center=None,
                         scaling=1):
    """Implements a far-field propagator in torch

    This accepts a torch tensor, where the last dimension
    represents the real and imaginary components of the wavefield,
    and returns the far-field propagated version of it using the provided
    geometrical information about the detector. It assumes that the
    propagation is purely far-field, without checking that the geometry
    is consistent with that assumption. Note that the pitch of the
    real space array is assumed to be consistent with the detector geometry,
    such that the pixel spacing on the detector corresponds to the full
    size covered by the wavefield array.

    It also assumes that the real space wavefront is stored in an array
    [i,j] where i corresponds to the y-axis and j corresponds to the
    x-axis, with the origin following the CS standard of being in the
    upper right.

    Args:
        wavefront (torch.Tensor) : The JxNxMx2 stack of complex wavefronts to be propagated
        detector_shape (array_like): The shape of the detector to simulate
        detector_center (array_like): Optional, the pixel (i,j) coordinates of the intersection of the detector with the forward propagation direction.
        scaling (int) : Default is 1, the downscaling to apply to the measured diffraction pattern
    Returns:
        torch.Tensor : The Jxdetector_shapex2 propagated wavefield
    """

    if detector_center is None:
        # Default is the exact center. This is subtly different from the
        # default for a shifted FFT, where for even-sized arrays, the
        # zero frequency pixel is placed at (shape-1)//2, not (shape-1)/2
        detector_center = (np.array(detector_shape)-np.array([1,1]))/2

    center = np.array(detector_center)
    # Split the center into a pixel and subpixel shift
    int_center = np.floor(center).astype(int)
    subpixel_shift = (center - int_center)

    # A selection pulling out the final area from the simulated
    # diffraction pattern
    # To be used as arr[sel[0]:sel[1],sel[2]:sel[3]]
    # Use the wavefront shape before downsampling
    wf_shape = tuple(wavefront.shape)
    sel = ((wf_shape[-3]-1)//2 - scaling*int_center[0],
           (wf_shape[-3]-1)//2 - scaling*int_center[0]
           + detector_shape[0]*scaling,
           (wf_shape[-2]-1)//2 - scaling*int_center[1],
           (wf_shape[-2]-1)//2 - scaling*int_center[1]
           + detector_shape[1]*scaling)
    # This generates a phase ramp to use for the final subpixel
    # shift.
    Is, Js = np.mgrid[0:wavefront.shape[-3],0:wavefront.shape[-2]]
    Is = Is - np.mean(Is)
    Js = Js - np.mean(Js)
    locs = np.stack((Is, Js), axis=-1)
    # Move from pixel to frequency units
    phase_ramp_freq = 2 * np.pi * subpixel_shift / wavefront.shape[-3:-1]
    phase_ramp = np.exp(-1j * np.dot(locs, phase_ramp_freq))
    phase_ramp = complex_to_torch(phase_ramp).to(device=wavefront.device,
                                                 dtype=wavefront.dtype)

    ramped_wavefront = cmult(phase_ramp[None,...], wavefront)

    sims = fftshift(fft(ifftshift(ramped_wavefront, dims=(-2,-3)),
                          2, normalized=True), dims=(-2,-3))

    return sims[:,sel[0]:sel[1],sel[2]:sel[3]]

def inverse_far_field(wavefront, detector_shape, detector_center=None,
                         scaling=1):
    if detector_center is None:
        # Default is the exact center. This is subtly different from the
        # default for a shifted FFT, where for even-sized arrays, the
        # zero frequency pixel is placed at (shape-1)//2, not (shape-1)/2
        detector_center = (np.array(detector_shape)-np.array([1,1]))/2


def inverse_near_field():
    pass



def near_field(wavefront, spacing, wavelength, z):
    """Implements an angular-spectrum based near-field propagator in torch

    This function is an angular-spectrum based near field
    propagator that will work on torch Tensors. The function is structured
    this way - to generate the propagator first - because the
    generation of the propagation mask is a bit expensive and if this
    propagator is used in a reconstruction program, then it will be best
    to calculate this mask once and close over it.
    The resulting function accepts an 3d torch tensor, where the
    last dimension represents the real and imaginary components of
    the wavefield, and returns the near-field propagated version of it.

    Args:
        wavefront (torch.Tensor) : The JxNxMx2 stack of complex wavefronts to be propagated
        spacing (iterable) : The pixel size in each dimension of the arrays to be propagated
        wavelength (float) : The wavelength of light to simulate propagation of
        z (float) : The distance to simulate propagation over
    Returns:
        function : A function to propagate a torch tensor.
    """

    ki = fftpack.fftfreq(shape[0],spacing[0])
    kj = fftpack.fftfreq(shape[1],spacing[1])
    Ki, Kj = np.meshgrid(ki,kj)
    propagator = np.exp(1j*np.sqrt((2*np.pi/wavelength)**2
                                - Ki**2 - Kj**2) * z)
    propagator = complex_to_float(propagator).astype(np.float32)
    propagator = t.from_numpy(propagator).cuda()

    return t.ifft(propagator * t.fft(wavefront,2),2)



def get_exit_waves(probe, object, translations):
    """Returns a stack of exit waves accounting for subpixel shifts

    This function returns a collection of exit waves, with the first
    dimension as the translation index and the final dimensions
    corresponding to the detector. The exit waves are calculated by
    shifting the object with each translation in turn, using linear
    interpolation.
    Args:
        probe (torch.Tensor) : An MxM probe function for the exit waves
        object (torch.Tensor) : The object function to be probed
        translations (torch.Tensor) : The Nx2 array of translations to simulate
    Returns:
        torch.Tensor : An NxMxM tensor of the calculated exit waves
    """

    # Separate the translations into a part that chooses the window
    # And a part that defines the windowing function
    integer_translations = t.floor(translations)
    subpixel_translations = translations - integer_translations
    integer_translations = integer_translations.to(dtype=t.int32)

    selections = []
    for tr, sp in zip(integer_translations,
                      subpixel_translations):

        sel00 = object[tr[0]:tr[0]+probe.shape[0],
                    tr[1]:tr[1]+probe.shape[1]]

        sel01 = object[tr[0]:tr[0]+probe.shape[0],
                    tr[1]+1:tr[1]+1+probe.shape[1]]

        sel10 = object[tr[0]+1:tr[0]+1+probe.shape[0],
                    tr[1]:tr[1]+probe.shape[1]]

        sel11 = object[tr[0]+1:tr[0]+1+probe.shape[0],
                    tr[1]+1:tr[1]+1+probe.shape[1]]

        selections.append(sel00 * (1-sp[0])*(1-sp[1]) + \
                          sel01 * (1-sp[0])*sp[1] + \
                          sel10 * sp[0]*(1-sp[1]) + \
                          sel11 * sp[0]*sp[1])

    return t.stack([cmult(probe,selection) for selection in selections])

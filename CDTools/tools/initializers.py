from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t

__all__ = ['exit_wave_geometry', 'gaussian']

from CDTools.tools import cmath
from scipy.fftpack import next_fast_len
import numpy as np


def exit_wave_geometry(det_basis, det_shape, wavelength, distance, center=None, opt_for_fft=True):
    """Returns an exit wave basis and a detector slice for the given detector geometry
    
    It takes in the parameters for a given detector - the basis defining
    the pixel pitch and the shape, as well as the wavelength and propagation
    distance. Optionally, it accepts a defined "center", the zero-frequency
    pixel's location. It then will automatically define a larger detector
    if necessary, define the exit wave basis associated with a far-field
    diffraction experiment, and return that basis, shape, and detector slice
    
    Args:
        det_basis (torch.Tensor) : The detector basis, as defined elsewhere
        det_shape (torch.Size) : The (i,j) shape of the detector
        wavelength (float) : The wavelength of light for the experiment, in m
        distance (float) : The sample-detector distance, in m
        center (torch.Tensor) : If defined, the location of the zero frequency pixel
        opt_for_fft (bool) : Default is true, whether to increase detector size to improve fft performance
    
    Returns:
        torch.Tensor : The exit wave basis
        torch.Tensor : The exit wave's shape
        tuple(slice) : The slice corresponding to the physical detector
    """
    det_shape = t.Tensor(tuple(det_shape))
    # First, set the center if it's not already specified
    # This definition matches the center pixel of an fftshifted array
    if center is None:
        center = det_shape // 2

    # Then, calculate the required detector size from the centering
    # This is a bit opaque but was worth doing accurately
    min_left = center * 2
    min_right = (det_shape - center) * 2 - 1
    full_shape = t.max(min_left,min_right).to(t.int32)
    if opt_for_fft:
        full_shape = t.Tensor([next_fast_len(dim) for dim in full_shape]).to(t.int32)
    # Then, generate a slice that pops the actual detector from the full
    # detector shape
    full_center = full_shape // 2
    det_slice = np.s_[int(full_center[0]-center[0]):
                      int(full_center[0]-center[0]+det_shape[0]),
                      int(full_center[1]-center[1]):
                      int(full_center[1]-center[1]+det_shape[1])]

    # Finally, generate the basis for the exit wave in real space
    # I believe this calculation is incorrect for non-rectangular
    # detectors, because the real space basis shoud be related to the
    # dual of the original basis. Leaving this for now since
    # non-rectangular detectors are not a pressing concern.
    basis_dirs = det_basis / t.norm(det_basis, dim=0)
    real_space_basis = basis_dirs * wavelength * distance / \
        (full_shape.to(t.float32) * t.norm(det_basis,dim=0))

    # Finally, convert the shape back to a torch.Size
    full_shape = t.Size([dim for dim in full_shape])
    
    return real_space_basis, full_shape, det_slice
                      
        

def gaussian(shape, amplitude, sigma, center = None):
    """Returns an array with a centered gaussian

    Takes in the shape, amplitude, and standard deviation of a gaussian
    and returns a torch tensor with values corresponding to a two-dimensional
    gaussian function

    Note that [0, 0] is taken to be at the upper left corner of the array.
    Default is centered at ((shape[0]-1)/2, (shape[1]-1)/2)) because x and y are zero-indexed.

    Args:
        shape (array_like) : A 1x2 array-like object specifying the dimensions of the output array in the form (i shape, j shape)
        amplitude (float or int): The amplitude the gaussian to simulate
        sigma (array_like): A 1x2 array-like object specifying the i- and j- standard deviation of the gaussian in the form (i stdev, j stdev)
        center (array_like) : Optional 1x2 array-like object specifying the location of the center of the gaussian (i center, j center)

    Returns:
        torch.Tensor : The real-valued gaussian array
    """
    if center is None:
        center = ((shape[0]-1)/2, (shape[1]-1)/2)
        
    i, j = np.mgrid[:shape[0], :shape[1]]
    result = amplitude*np.exp(-( (i-center[0])**2 / (2 * sigma[0]**2) )
                              -( (j-center[1])**2 / (2 * sigma[1]**2) ))
    return cmath.complex_to_torch(result)

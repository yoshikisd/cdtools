from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t

all = ['gaussian']

from CDTools.tools import cmath


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

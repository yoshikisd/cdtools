from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t

all = ['gaussian']


def gaussian(shape, amplitude, sigma, center = None):
    """Returns an array with a centered gaussian

    Takes in the shape, amplitude, and standard deviation of a gaussian
    and returns an array with values corresponding to a two-dimensional gaussian function
        z = amplitude*exp(-(x-center[0])**2/sigma[0]**2+(y-center[1])**2/sigma[1]**2)
    Note that [0, 0] is taken to be at the upper left corner of the array.
    Default is centered at (shape[0]/2, shape[1]/2).

    Args:
        shape (array_like) : A 1x2 array-like object specifying the dimensions of the output array
        amplitude (float or int): The amplitude the gaussian to simulate
        sigma (array_like): A 1x2 array-like object specifying the x- and y- standard deviation of the gaussian
        center (array_like) : Optional 1x2 array-like object specifying the location of the center of the gaussian

    Returns:
        torch.Tensor : The real-valued gaussian array
    """
    x, y = np.meshgrid(shape)
    return x

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
    Default is centered at ((shape[0]-1)/2, (shape[1]-1)/2)) because x and y are zero-indexed.

    Args:
        shape (array_like) : A 1x2 array-like object specifying the dimensions of the output array in the form (y shape, x shape)
        amplitude (float or int): The amplitude the gaussian to simulate
        sigma (array_like): A 1x2 array-like object specifying the x- and y- standard deviation of the gaussian in the form (y stdev, y stdev)
        center (array_like) : Optional 1x2 array-like object specifying the location of the center of the gaussian (y center, x center)

    Returns:
        numpy.array : The real-valued gaussian array
    """
    if center is None:
        center = ((shape[0]-1)/2, (shape[1]-1)/2)
    y, x = np.mgrid[:shape[0], :shape[1]]
    return amplitude*np.exp(-((x-center[1])/sigma[1])**2-((y-center[0])/sigma[0])**2)

from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from CDTools.tools import cmath


def centroid(im, dims=2):
    """Returns the centroid of an image or a stack of images
    
    By default, the last two dimensions are used in the calculation
    and the remainder of the dimensions are passed through.
    
    Beware that the meaning of the centroid is not well defined if your
    image contains values less than 0

    Args:
        im (t.Tensor) : An image or stack of images to calculate from
        dims (int) : Default 2, how many trailing dimensions to calculate for

    Returns:
        t.Tensor : An (i,j) index or stack of indices
    """
    indices = (t.arange(im.shape[-dims+i]).to(t.float32) for i in range(dims))
    indices = t.meshgrid(*indices)

    use_dims = [-dims+i for i in range(dims)]
    divisor = t.sum(im, dim=use_dims)
    centroids = [t.sum(index * im, dim=use_dims) / divisor
                 for index in indices]

    return t.stack(centroids,dim=-1)
    



def centroid_sq(im, dims=2, comp=False):
    """Returns the centroid of the square of an image or stack of images
    
    By default, the last two dimensions are used in the calculation
    and the remainder of the dimensions are passed through.

    If the "comp" flag is set, it will be assumed that the last dimension
    represents the real and imaginary part of a complex number, and the
    centroid will be calculated for the magnitude squared of those numbers
    
    Args:
        im (t.Tensor) : An image or stack of images to calculate from
        dims (int) : Default 2, how many trailing dimensions to calculate for
        comp (bool) : Default is False, whether the data represents complex numbers
    Returns:
        t.Tensor : An (i,j) index or stack of indices
    """
    if comp:
        im_sq = cmath.cabssq(im)
    else:
        im_sq = im**2

    return centroid(im_sq, dims=dims)

    

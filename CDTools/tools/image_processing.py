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


def find_subpixel_shift(im1, im2, search_around=(0,0), resolution=10):
    """Calculates the subpixel shift between two images by maximizing the autocorrelation

    This function only searches in a 2 pixel by 2 pixel box around the
    specified search_around parameter. The calculation is done using the
    approach outlined in "Efficient subpixel image registration algorithms",
    Optics Express (2008) by Manual Guizar-Sicarios et al.

    Args:
        im1 (t.Tensor): The first real or complex-valued torch tensor
        im2 (t.Tensor): The second real or complex-valued torch tensor
        search_around (array_like) : Default (0,0), the shift to search in the vicinity of
        resolution (int): Default is 10, the resolution to calculate to in units of 1/n
    """
    pass

    
def find_pixel_shift(im1, im2):
    """Calculates the integer pixel shift between two images by maximizing the autocorrelation

    This function simply takes the circular correlation with an FFT and
    returns the position of the maximum of that correlation. This corresponds
    to the amount that im1 would have to be shifted by to line up best with
    im2

    Args:
        im1 (t.Tensor): The first real or complex-valued torch tensor
        im2 (t.Tensor): The second real or complex-valued torch tensor
        search_around (array_like) : Default (0,0), the shift to search in the vicinity of
        resolution (int): Default is 10, the resolution to calculate to in units of 1/n
    """
    # If last dimension is not 2, then convert to a complex tensor now
    if im1.shape[-1] != 2:
        im1 = t.stack((im1,t.zeros_like(im1)),dim=-1)
    if im2.shape[-1] != 2:
        im2 = t.stack((im2,t.zeros_like(im2)),dim=-1)

    
    cor = cmath.cabs(t.ifft(cmath.cmult(t.fft(im1,2),
                                        cmath.cconj(t.fft(im2,2))),2))
    
    sh = t.tensor(cor.shape).to(device=im1.device)
    cormax = t.tensor([t.argmax(cor) // sh[1],
                       t.argmax(cor) % sh[1]]).to(device=im1.device)
    return (cormax + sh // 2) % sh - sh//2



def find_shift(im1, im2, resolution=10):
    """Calculates the shift between two images by maximizing the autocorrelation

    This function starts by calculating the maximum shift to integer
    pixel resolution, and then searchers the nearby area to calculate a
    subpixel shift

    Args:
        im1 (t.Tensor): The first real or complex-valued torch tensor
        im2 (t.Tensor): The second real or complex-valued torch tensor
        resolution (int): Default is 10, the resolution to calculate to in units of 1/n
    """
    integer_shift = find_pixel_shift(im1,im2)
    subpixel_shift = find_subpixel_shift(im1, im2, search_around=integer_shift,
                                         resolution=resolution)

    return subpixel_shift
    

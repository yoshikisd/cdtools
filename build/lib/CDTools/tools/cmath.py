"""Contains basic functions for dealing with complex numbers in pytorch.

Since pytorch doesn't have built-in support for complex numbers, but the
fast fourier transforms in pytorch assume a specific format for complex
arrays, this module uses that format to store complex numbers. It exposes
functions for converting between complex numpy arrays and torch tensors
stored in that format, as well as basic complex math operations implemented
on the torch tensors
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t


__all__ = ['complex_to_torch','torch_to_complex','cabssq','cabs','cconj',
           'cmult', 'cdiv', 'cphase', 'fftshift', 'ifftshift']


#
# These define the conversions to and from this format
#

def complex_to_torch(x):
    """Maps a complex numpy array to a torch tensor

    Pytorch uses tensors with a final dimension of 2 to represent
    complex numbers. This maps a complex type numpy array to a torch
    tensor following this convention

    Args:
        x (array_like): A numpy array to convert

    Returns:
        torch.Tensor : A torch tensor representation of that array

    """
    return t.from_numpy(np.stack((np.real(x),np.imag(x)),axis=-1))


def torch_to_complex(x):
    """Maps a torch tensor to the a complex numpy array

    Pytorch uses tensors with a final dimension of 2 to represent
    complex numbers. This maps a torch tensor following that convention
    to the appropriate numpy complex array

    Args:
        x (torch.Tensor): A tensor to convert

    Returns:
        np.array : A complex typed numpy array corresponding to the input

    """
    x = np.array(x)
    x = x[...,0] + x[...,1] * 1j
    return x


#
# And these define the basic operations on these arrays. Note that
# multiplication between a complex valued and real valued pytorch
# tensor will proceed as expected because of torch's broadcasting
# and thus doesn't need it's own function
#

def cabssq(a):
    """Returns the square of the absolute value of a complex torch tensor

    Pytorch uses tensors with a final dimension of 2 to represent
    complex numbers. This calculates the elementwise absolute value
    squared of any toch tensor following that standard.

    Args:
        x (torch.Tensor): An input tensor

    Returns:
        array_like : A tensor storing the elementwise absolute value squared

    """
    return a[...,0]**2 + a[...,1]**2


def cabs(a):
    """Returns the absolute value of a complex torch tensor

    Pytorch uses tensors with a final dimension of 2 to represent
    complex numbers. This calculates the elementwise absolute value
    of any torch tensor following that standard.

    Args:
        x (torch.Tensor): An input tensor

    Returns:
        array_like : A tensor storing the elementwise absolute value

    """
    return t.sqrt(cabssq(a))


def cphase(a):
    """Returns the complex conjugate of a complex torch tensor

    Pytorch uses tensors with a final dimension of 2 to represent
    complex numbers. This calculates the elementwise complex phase
    of any torch tensor following that standard.

    Args:
        x (torch.Tensor): An input tensor

    Returns:
        array_like : A tensor storing the elementwise phase

    """
    return t.atan2(a[...,1],a[...,0])
    
    
def cconj(a):
    """Returns the complex conjugate of a complex torch tensor

    Pytorch uses tensors with a final dimension of 2 to represent
    complex numbers. This calculates the elementwise complex conjugate
    of any torch tensor following that standard.

    Args:
        x (torch.Tensor): An input tensor

    Returns:
        array_like : A tensor storing the elementwise complex conjugate

    """
    return t.stack((a[...,0],-a[...,1]),dim=-1)



def cmult(a,b):
    """Returns the complex product of two torch tensors

    Pytorch uses tensors with a final dimension of 2 to represent
    complex numbers. This calculates the elementwise product
    of two torch tensors following that standard.

    Args:
        a (torch.Tensor): An input tensor
        b (torch.Tensor): A second input tensor

    Returns:
        torch.Tensor : A tensor storing the elementwise product

    """
    
    real = a[...,0] * b[...,0] - a[...,1] * b[...,1]
    imag = a[...,0] * b[...,1] + a[...,1] * b[...,0]
    return t.stack((real,imag),dim=-1)


def cdiv(a,b):
    """Returns the complex quotient of two torch tensors

    Pytorch uses tensors with a final dimension of 2 to represent
    complex numbers. This calculates the elementwise quotient
    of two torch tensors following that standard.

    Args:
        a (torch.Tensor): An input tensor
        b (torch.Tensor): A second input tensor

    Returns:
        torch.Tensor : A tensor storing the elementwise complex quotient

    """
    return cmult(a, cconj(b)) / t.unsqueeze(cabssq(b),-1)


    
#
# Not entirely sure if these belong here, but heck with it.
# We just need the ability to do fftshifts
#


def fftshift(array,dims=None):
    """Drop-in torch replacement for scipy.fftpack.fftshift
    
    This maps a tensor, assumed to be the output of a fast Fourier
    transform, into a tensor whose zero-frequency element is at the
    center of the tensor instead of the start. It will by default shift
    every dimension in the tensor but the last (which is assumed to
    represent the complex number and be of dimension 2), but can shift
    any arbitrary set of dimensions.
    
    Args:
        array (torch.Tensor) : An array of data to be fftshifted
        dims (iterable) : A list of all dimensions to shift

    Returns:
        torch.Tensor : fftshifted tensor
    
    """
    
    if dims is None:
        dims=list(range(array.dim()))[:-1]
    for dim in dims:
        length = array.size()[dim]
        cut_to = (length + 1) // 2
        cut_len = length - cut_to
        array = t.cat((array.narrow(dim,cut_to,cut_len),
                       array.narrow(dim,0,cut_to)), dim)
    return array



def ifftshift(array,dims=None):
    """Drop-in torch replacement for scipy.fftpack.iftshift
    
    This maps a tensor, assumed to be the shifted output of a fast
    Fourier transform, into a tensor whose zero-frequency element is
    back at the start of the tensor instead of the center. It is the
    inverse of the fftshift operator. It will by default shift
    every dimension in the tensor but the last (which is assumed to
    represent the complex number and be of dimension 2), but can shift
    any arbitrary set of dimensions.
    
    Args:
        array (torch.Tensor) : An array of data to be ifftshifted
        dims (iterable) : A list of all dimensions to shift

    Returns:
        torch.Tensor : ifftshifted tensor
    
    """
    
    if dims is None:
        dims=list(range(array.dim()))[:-1]
    for dim in dims:
        length = array.size()[dim]
        cut_to = length // 2
        cut_len = length - cut_to
        
        array = t.cat((array.narrow(dim,cut_to,cut_len),
                       array.narrow(dim,0,cut_to)), dim)
    return array



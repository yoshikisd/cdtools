from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
import numpy as np
import torch as t
from scipy.fftpack import fftshift, ifftshift

def test_complex_to_torch():
    arr = np.random.rand(100,4) + 1j * np.random.rand(100,4)
    tensor = cmath.complex_to_torch(arr)
    assert np.allclose(tensor[:,:,0].numpy(),np.real(arr))
    assert np.allclose(tensor[:,:,1].numpy(),np.imag(arr))

    
def test_torch_to_complex():
    tensor = t.rand(100,4,2)
    arr = cmath.torch_to_complex(tensor)
    assert np.allclose(tensor[:,:,0].numpy(),np.real(arr))
    assert np.allclose(tensor[:,:,1].numpy(),np.imag(arr))


def test_cabssq():
    arr = np.random.rand(56,23,2) + 1j *  np.random.rand(56,23,2)
    cabssq = cmath.cabssq(cmath.complex_to_torch(arr))
    assert np.allclose(cabssq.numpy(),np.abs(arr)**2)


def test_cabs():
    arr = np.random.rand(2,3,4,5) + 1j * np.random.rand(2,3,4,5)
    cabs = cmath.cabs(cmath.complex_to_torch(arr))
    assert np.allclose(cabs.numpy(),np.abs(arr))

    
def test_cconj():
    arr = np.random.rand(50) + 1j * np.random.rand(50)
    cconj = cmath.cconj(cmath.complex_to_torch(arr))
    assert np.allclose(cmath.torch_to_complex(cconj), np.conj(arr))

    
def test_cmult():
    arr1 = np.random.rand(50) + 1j * np.random.rand(50)
    arr2 = np.random.rand(50) + 1j * np.random.rand(50)
    mult = cmath.cmult(cmath.complex_to_torch(arr1),
                       cmath.complex_to_torch(arr2))
    assert np.allclose(cmath.torch_to_complex(mult),arr1*arr2)

    
def test_cdiv():
    arr1 = np.random.rand(50) + 1j * np.random.rand(50)
    arr2 = np.random.rand(50) + 1j * np.random.rand(50)
    div = cmath.cdiv(cmath.complex_to_torch(arr1),
                     cmath.complex_to_torch(arr2))
    assert np.allclose(cmath.torch_to_complex(div),arr1 / arr2)

    
def test_cphase():
    arr = np.random.rand(50) + 1j * np.random.rand(50)
    cphase = cmath.cphase(cmath.complex_to_torch(arr))
    assert np.allclose(cphase.numpy(), np.angle(arr))


def test_scalars():
    arr1 = np.random.rand(1) + 1j * np.random.rand(1)
    arr2 = np.random.rand(1) + 1j * np.random.rand(1)
    div = cmath.cdiv(cmath.complex_to_torch(arr1),
                     cmath.complex_to_torch(arr2))
    assert np.allclose(cmath.torch_to_complex(div),arr1 / arr2)

def test_fftshift():
    #1D, even
    arr = np.random.rand(300) + 1j * np.random.rand(300)
    shifted = cmath.fftshift(cmath.complex_to_torch(arr))
    assert np.allclose(fftshift(arr),
                       cmath.torch_to_complex(shifted))
    #1D, odd
    arr = np.random.rand(301) + 1j * np.random.rand(301)
    shifted = cmath.fftshift(cmath.complex_to_torch(arr))
    assert np.allclose(fftshift(arr),
                       cmath.torch_to_complex(shifted))

    #2D
    arr = np.random.rand(20,21) + 1j * np.random.rand(20,21)
    shifted = cmath.fftshift(cmath.complex_to_torch(arr))
    assert np.allclose(fftshift(arr),
                       cmath.torch_to_complex(shifted))
    #3D
    arr = np.random.rand(15,16,17) + 1j * np.random.rand(15,16,17)
    shifted = cmath.fftshift(cmath.complex_to_torch(arr))
    assert np.allclose(fftshift(arr),
                       cmath.torch_to_complex(shifted))

    #3D, choosing specific axes
    arr = np.random.rand(15,16,17) + 1j * np.random.rand(15,16,17)
    shifted = cmath.fftshift(cmath.complex_to_torch(arr),dims=(0,1))
    assert np.allclose(fftshift(arr,axes=(0,1)),
                       cmath.torch_to_complex(shifted))


def test_ifftshift():
    #1D, even
    arr = np.random.rand(300) + 1j * np.random.rand(300)
    shifted = cmath.ifftshift(cmath.complex_to_torch(arr))
    assert np.allclose(ifftshift(arr),
                       cmath.torch_to_complex(shifted))
    #1D, odd
    arr = np.random.rand(301) + 1j * np.random.rand(301)
    shifted = cmath.ifftshift(cmath.complex_to_torch(arr))
    assert np.allclose(ifftshift(arr),
                       cmath.torch_to_complex(shifted))

    #2D
    arr = np.random.rand(20,21) + 1j * np.random.rand(20,21)
    shifted = cmath.ifftshift(cmath.complex_to_torch(arr))
    assert np.allclose(ifftshift(arr),
                       cmath.torch_to_complex(shifted))
    #3D
    arr = np.random.rand(15,16,17) + 1j * np.random.rand(15,16,17)
    shifted = cmath.ifftshift(cmath.complex_to_torch(arr))
    assert np.allclose(ifftshift(arr),
                       cmath.torch_to_complex(shifted))

    #3D, choosing specific axes
    arr = np.random.rand(15,16,17) + 1j * np.random.rand(15,16,17)
    shifted = cmath.ifftshift(cmath.complex_to_torch(arr),dims=(0,1))
    assert np.allclose(ifftshift(arr,axes=(0,1)),
                       cmath.torch_to_complex(shifted))


def test_expi():
    phases = np.random.rand(20,70) * 2 * np.pi
    np_result = np.exp(1j * phases)

    torch_result = cmath.expi(t.Tensor(phases))

    assert np.allclose(np_result, cmath.torch_to_complex(torch_result))

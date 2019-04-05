from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
from CDTools.tools import initializers
from CDTools.tools import propagators

import numpy as np
import torch as t
import pytest
import scipy.misc
from scipy.fftpack import fftshift, ifftshift



@pytest.fixture(scope='module')
def exit_waves_1():
    # Import scipy test image and add a random phase
    obj = scipy.misc.ascent()[0:64,0:64].astype(np.complex128)
    arr = np.random.random_sample((64,64))
    obj *= (arr+(1-arr**2)**.5*1j)
    obj = cmath.complex_to_torch(obj)

    # Construct wavefront from image
    probe = initializers.gaussian([64, 64], [5, 5], amplitude=1e3)
    return cmath.cmult(probe,obj)

    

def test_far_field(exit_waves_1):
    # Far field diffraction patterns calculated by numpy with zero frequency in center
    np_result = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cmath.torch_to_complex(exit_waves_1)),norm='ortho'))

    assert(np.allclose(np_result, cmath.torch_to_complex(propagators.far_field(exit_waves_1))))



def test_inverse_far_field(exit_waves_1):
    # We want the inverse far field to map back to the exit waves with no intensity corrections
    # Far field result for exit waves calculated with numpy
    far_field_np_result = cmath.complex_to_torch(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(cmath.torch_to_complex(exit_waves_1)),norm='ortho')))

    assert(np.allclose(exit_waves_1, propagators.inverse_far_field(far_field_np_result)))



def test_near_field():

    # The strategy is to compare the propagation of a gaussian beam to
    # the propagation in the paraxial approximation.
    
    x = (np.arange(800) - 400) * 1.5e-9
    y = (np.arange(1200) - 600) * 1e-9
    Ys,Xs = np.meshgrid(y,x)
    Rs = np.sqrt(Xs**2+Ys**2)

    wavelength = 3e-9 #nm
    sigma = 20e-9 #nm
    z = 1000e-9 #nm

    k = 2 * np.pi / wavelength
    w0 = np.sqrt(2)*sigma
    zr = np.pi * w0**2 / wavelength
    wz = w0 * np.sqrt(1 + (z / zr)**2)
    Rz =  z * (1 + (zr / z)**2)
    
    E0 = np.exp(-Rs**2 / w0**2)

    # The analytical expression for propagation of a gaussian beam in the
    # paraxial approx
    Ez = w0 / wz * np.exp(-Rs**2 / wz**2) * np.exp(-1j * k * ( z + Rs**2 / (2 * Rz)) + 1j * np.arctan(z / zr))

    asp = propagators.generate_angular_spectrum_propagator(
        E0.shape,(1.5e-9,1e-9),wavelength,z,dtype=t.float64)
    
    Ez_t = propagators.near_field(cmath.complex_to_torch(E0),asp)
    Ez_t = cmath.torch_to_complex(Ez_t)
    
    # Check for at least 10^-3 relative accuracy in this scenario
    assert np.max(np.abs(Ez-Ez_t)) < 1e-3 * np.max(np.abs(Ez))


    Emz = np.conj(Ez)

    Emz_t = propagators.inverse_near_field(cmath.complex_to_torch(E0),asp)
    Emz_t = cmath.torch_to_complex(Emz_t)    

    # Again, 10^-3 is about all the accuracy we can expect
    assert np.max(np.abs(Emz-Emz_t)) < 1e-3 * np.max(np.abs(Emz))

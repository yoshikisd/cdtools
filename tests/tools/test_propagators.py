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
    object = scipy.misc.ascent()[0:64,0:64].astype(np.complex128)
    arr = np.random.random_sample((64,64))
    object *= (arr+(1-arr**2)**.5*1j)

    # Construct wavefront from image
    probe = initializers.gaussian([64, 64], 1e3, [5, 5])*(1+1j)
    return cmath.complex_to_torch(probe*object)


def test_far_field(exit_waves_1):
    # Far field diffraction patterns calculated by numpy with zero frequency in center
    np_result = np.fft.fftshift(np.fft.fft2(cmath.torch_to_complex(exit_waves_1)))

    assert(np.allclose(np_result, cmath.torch_to_complex(propagators.far_field(exit_waves_1))))


def test_inverse_far_field(exit_waves_1):
    # We want the inverse far field to map back to the exit waves with no intensity corrections
    np_result = exit_waves_1
    # Far field result for exit waves calculated with numpy
    far_field_np_result = cmath.complex_to_torch(np.fft.fftshift(np.fft.fft2(cmath.torch_to_complex(exit_waves_1))))

    assert(np.allclose(np_result, propagators.inverse_far_field(far_field_np_result)))



def test_near_field(exit_waves_1):
    pass


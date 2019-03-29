from __future__ import division, print_function, absolute_import

from CDTools.tools import measurements
from CDTools.tools import cmath
import torch as t
import numpy as np
import pytest


def test_intensity():
    wavefield = t.rand((10,10,2))
    np_result = np.abs(cmath.torch_to_complex(wavefield))**2
    assert t.allclose(measurements.intensity(wavefield),
                      t.tensor(np_result))

    det_slice = np.s_[3:,5:8]
    assert t.allclose(measurements.intensity(wavefield,det_slice),
                      t.tensor(np_result[det_slice]))



def test_incoherent_sum():
    wavefields = t.rand((4,10,10,2))
    np_result = np.sum(np.abs(cmath.torch_to_complex(wavefields))**2,axis=0)
    assert t.allclose(measurements.incoherent_sum(wavefields),
                      t.tensor(np_result))

    det_slice = np.s_[3:,5:8]
    assert t.allclose(measurements.incoherent_sum(wavefields,det_slice),
                      t.tensor(np_result[det_slice]))



def test_quadratic_background():
    # test with intensity
    wavefield = t.rand((10,10,2))
    background = t.rand((10,10))
    np_result = np.abs(cmath.torch_to_complex(wavefield))**2 + background.numpy()**2
    det_slice = np.s_[3:,5:8]

    result = measurements.quadratic_background(wavefield,background,
                                               detector_slice=det_slice,
                                               measurement=measurements.intensity)
    assert t.allclose(result, t.tensor(np_result[det_slice]))
   
    # test with incoherent sum but no slice
    wavefields = t.rand((4,10,10,2))
    np_result = np.sum(np.abs(cmath.torch_to_complex(wavefields))**2,axis=0)
    np_result += background.numpy()**2
    result = measurements.quadratic_background(wavefields, background,
                                               measurement=measurements.incoherent_sum)
    assert t.allclose(result, t.tensor(np_result))

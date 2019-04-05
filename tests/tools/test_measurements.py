from __future__ import division, print_function, absolute_import

from CDTools.tools import measurements
from CDTools.tools import cmath
import torch as t
import numpy as np
import pytest


def test_intensity():
    wavefields = t.rand((5,10,10,2))
    np_result = np.abs(cmath.torch_to_complex(wavefields))**2
    assert t.allclose(measurements.intensity(wavefields),
                      t.tensor(np_result))

    # Test single field case
    assert t.allclose(measurements.intensity(wavefields[0]),
                      t.tensor(np_result[0]))

    
    det_slice = np.s_[3:,5:8]
    assert t.allclose(measurements.intensity(wavefields,det_slice),
                      t.tensor(np_result[(np.s_[:],)+det_slice]))
    
    # Test single field case
    assert t.allclose(measurements.intensity(wavefields[0],det_slice),
                      t.tensor(np_result[0][det_slice]))


def test_incoherent_sum():
    wavefields = t.rand((5,4,10,10,2))
    np_result = np.sum(np.abs(cmath.torch_to_complex(wavefields))**2,axis=1)
    assert t.allclose(measurements.incoherent_sum(wavefields),
                      t.tensor(np_result))
    # Test single field case
    assert t.allclose(measurements.incoherent_sum(wavefields[0]),
                      t.tensor(np_result[0]))

    
    det_slice = np.s_[3:,5:8]
    assert t.allclose(measurements.incoherent_sum(wavefields,det_slice),
                      t.tensor(np_result[(np.s_[:],)+det_slice]))
    # Test single field case
    assert t.allclose(measurements.incoherent_sum(wavefields[0],det_slice),
                      t.tensor(np_result[0][det_slice]))


def test_quadratic_background():
    # test with intensity
    wavefields = t.rand((5,10,10,2))
    background = t.rand((10,10))
    np_result = np.abs(cmath.torch_to_complex(wavefields))**2 + background.numpy()**2
    det_slice = np.s_[3:,5:8]

    result = measurements.quadratic_background(wavefields,background[det_slice],
                                               detector_slice=det_slice,
                                               measurement=measurements.intensity)
    assert t.allclose(result, t.tensor(np_result[(np.s_[:],)+det_slice]))
    
    
    # test with incoherent sum but no slice and no stack
    wavefields = t.rand((4,10,10,2))
    np_result = np.sum(np.abs(cmath.torch_to_complex(wavefields))**2,axis=0)
    np_result += background.numpy()**2
    result = measurements.quadratic_background(wavefields, background,
                                               measurement=measurements.incoherent_sum)
    assert t.allclose(result, t.tensor(np_result))

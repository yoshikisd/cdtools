from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
from CDTools.tools import interactions
import numpy as np
import torch as t
import pytest


# Have a random probe and a random object and test the two
# functions for a variety of overlaps

# Also I want a probe that's just a single pixel
#

@pytest.fixture(scope='module')
def random_probe():
    return np.random.rand(256,256) * np.exp(2j * np.pi * np.random.rand(256,256)) 

@pytest.fixture(scope='module')
def random_obj():
    return np.random.rand(900,900) * np.exp(2j * np.pi * np.random.rand(900,900)) 

@pytest.fixture(scope='module')
def single_pixel_probe(scope='module'):
    probe = np.zeros((256,256), dtype=np.complex128)
    probe[128,128] = 1
    return probe


def test_ptycho_2D_round(random_probe, random_obj):
    # Test a stack of images
    translations = np.random.rand(10,2) * 500
    exit_waves_np = [random_probe * \
                     random_obj[tr[0]:tr[0]+random_probe.shape[0],
                                tr[1]:tr[1]+random_probe.shape[1]] for
                     tr in np.round(translations).astype(int)]
    exit_waves_t = interactions.ptycho_2D_round(cmath.complex_to_torch(random_probe),
                                                cmath.complex_to_torch(random_obj),
                                                t.tensor(translations))
    assert np.allclose(cmath.torch_to_complex(exit_waves_t), exit_waves_np)

    # Test the single wave case
    exit_wave_t = interactions.ptycho_2D_round(cmath.complex_to_torch(random_probe),
                                                cmath.complex_to_torch(random_obj),
                                                t.tensor(translations[0]))
    assert np.allclose(cmath.torch_to_complex(exit_wave_t), exit_waves_np[0])



def test_ptycho_2D_linear(single_pixel_probe, random_obj):

    # For this one, I just want to check one translation, but
    # I need to check both formats
    translations = np.array([[46.7,53.2]])
    translation = np.array([46.7,53.2])
     
    exit_waves_probe = interactions.ptycho_2D_linear(
        cmath.complex_to_torch(single_pixel_probe),
        cmath.complex_to_torch(random_obj),
        t.tensor(translations),
        shift_probe=True)

    exit_wave_probe = interactions.ptycho_2D_linear(
        cmath.complex_to_torch(single_pixel_probe),
        cmath.complex_to_torch(random_obj),
        t.tensor(translation),
        shift_probe=True)

    # Check that the outputs match
    assert t.allclose(exit_waves_probe[0],exit_wave_probe)

    
    exit_waves_obj = interactions.ptycho_2D_linear(
        cmath.complex_to_torch(single_pixel_probe),
        cmath.complex_to_torch(random_obj),
        t.tensor(translations),
        shift_probe=False)

    exit_wave_obj = interactions.ptycho_2D_linear(
        cmath.complex_to_torch(single_pixel_probe),
        cmath.complex_to_torch(random_obj),
        t.tensor(translation),
        shift_probe=False)

    # Check that the outputs match
    assert t.allclose(exit_waves_obj[0],exit_wave_obj)

    # For the shifted probe, we should find 4 pixels with intensity
    exit_waves_probe = cmath.torch_to_complex(exit_waves_probe)[0]

    probe_shift = np.array([[0.3*0.8,0.3*0.2],
                            [0.7*0.8,0.7*0.2]])
    obj_section = random_obj[128+46:128+48,
                             128+53:128+55]
    exit_section = exit_waves_probe[128:130,128:130]
    assert np.allclose(probe_shift * obj_section, exit_section)
    
    # For the shifted obj, we should find one pixel with intensity
    exit_waves_obj = cmath.torch_to_complex(exit_waves_obj)[0]
    obj_shift = np.array([[0.3*0.8,0.3*0.2],
                            [0.7*0.8,0.7*0.2]])
    obj_section = random_obj[128+46:128+48,
                             128+53:128+55]
    exit_pixel = exit_waves_obj[128,128]
    assert np.isclose(np.sum(obj_shift * obj_section),exit_pixel)
    
    # Test for a single translation
    

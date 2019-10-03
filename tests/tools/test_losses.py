from __future__ import division, print_function, absolute_import

from CDTools.tools import losses
from CDTools.tools import cmath
import numpy as np
import torch as t


# The idea here is to use a simple numpy calculation of the various
# objective functions to check the torch implementations and make sure
# that any optimizations in the future don't change the results

def test_amplitude_mse():
    # Make some fake data
    data = np.random.rand(10,100,100)
    # And add some noise to it
    sim = data + 0.1 * np.random.rand(10,100,100)
    # and define a simple mask that needs to be broadcast
    mask = (np.random.rand(100,100) > 0.1).astype(np.bool)

    # First, test without a mask
    np_result = np.sum((np.sqrt(data) - np.sqrt(sim))**2)
    #np_result /=  data.size
    torch_result = losses.amplitude_mse(t.from_numpy(data),t.from_numpy(sim))
    assert np.isclose(np_result, np.take(torch_result.numpy(),0))

    # Then, test with a mask
    np_result = np.sum(mask * (np.sqrt(data) - np.sqrt(sim))**2)
    #np_result /=  np.count_nonzero(mask * np.ones_like(data))
    torch_result = losses.amplitude_mse(t.from_numpy(data),t.from_numpy(sim),
                         mask = t.from_numpy(mask))
    assert np.isclose(np_result, np.take(torch_result.numpy(),0))


def test_intensity_mse():
    # Make some fake data
    data = np.random.rand(10,100,100)
    # And add some noise to it
    sim = data + 0.1 * np.random.rand(10,100,100)
    # and define a simple mask that needs to be broadcast
    mask = (np.random.rand(100,100) > 0.1).astype(np.bool)


    # First, test without a mask
    np_result = np.sum((data - sim)**2)
    np_result /=  data.size
    torch_result = losses.intensity_mse(t.from_numpy(data),t.from_numpy(sim))
    assert np.isclose(np_result, np.take(torch_result.numpy(),0))

    # Then, test with a mask
    np_result = np.sum(mask * (data - sim)**2)
    np_result /=  np.count_nonzero(mask * np.ones_like(data))
    torch_result = losses.intensity_mse(t.from_numpy(data),t.from_numpy(sim),
                         mask = t.from_numpy(mask))
    assert np.isclose(np_result, np.take(torch_result.numpy(),0))
    

def test_poisson_ml():
    # Make some fake data
    data = np.random.rand(10,100,100)
    # And add some noise to it
    sim = data + 0.1 * np.random.rand(10,100,100)
    # and define a simple mask that needs to be broadcast
    mask = (np.random.rand(100,100) > 0.1).astype(np.bool)


    # First, test without a mask
    np_result = np.sum(sim - data * np.log(sim))
    np_result /=  data.size
    torch_result = losses.poisson_nll(t.from_numpy(data),t.from_numpy(sim))
    assert np.isclose(np_result, np.take(torch_result.numpy(),0))

    # Then, test with a mask
    np_result = np.sum(mask * (sim - data * np.log(sim)))
    np_result /=  np.count_nonzero(mask * np.ones_like(data))
    torch_result = losses.poisson_nll(t.from_numpy(data),t.from_numpy(sim),
                         mask = t.from_numpy(mask))
    assert np.isclose(np_result, np.take(torch_result.numpy(),0))




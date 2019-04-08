from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
from CDTools.tools import plotting
from CDTools.tools import initializers
import numpy as np
import pytest
import torch as t
import scipy.misc
import matplotlib.pyplot as plt

def test_plot_1d(show_plot):
    # Plot simple linear scatter plot
    arr = np.arange(10)
    plotting.plot_1d(arr, title = 'Linear Plot')
    if show_plot:
        plt.show()

def test_plot_amplitude(show_plot):
    # Test with tensor
    im = cmath.complex_to_torch(scipy.misc.ascent().astype(np.float64))
    plotting.plot_amplitude(im, basis = np.array([[1,1], [1,1], [0,0]]), title = 'Test Amplitude')
    if show_plot:
        plt.show()

    # Test with numpy array
    im = scipy.misc.ascent().astype(np.complex128)
    plotting.plot_amplitude(im, title = 'Test Amplitude')
    if show_plot:
        plt.show()


def test_plot_phase(show_plot):
    # Test with tensor
    im = initializers.gaussian([512, 512], [200,200], amplitude=100, curvature=[.1,.1])
    plotting.plot_phase(im, title = 'Test Phase')
    if show_plot:
        plt.show()

    # Test with numpy array
    im = cmath.torch_to_complex(initializers.gaussian([512, 512], [200,200], amplitude=100, curvature=[.1,.1]))
    plotting.plot_phase(im, title = 'Test Phase', basis = np.array([[1,1], [1,1], [0,0]]))
    if show_plot:
        plt.show()

def test_plot_colorize(show_plot):
    # Test with tensor
    gaussian = initializers.gaussian([512, 512], [200,200], amplitude=100, curvature=[.1,.1])
    im = cmath.cmult(gaussian, cmath.complex_to_torch(scipy.misc.ascent().astype(np.float64)))
    plotting.plot_colorized(im, title = 'Test Colorize', basis = np.array([[1,1], [1,1], [0,0]]))
    if show_plot:
        plt.show()

    # Test with numpy array
    gaussian = initializers.gaussian([512, 512], [200,200], amplitude=100, curvature=[.1,.1])
    im = cmath.torch_to_complex(cmath.cmult(gaussian, cmath.complex_to_torch(scipy.misc.ascent().astype(np.float64))))
    plotting.plot_colorized(im, title = 'Test Colorize')
    if show_plot:
        plt.show()

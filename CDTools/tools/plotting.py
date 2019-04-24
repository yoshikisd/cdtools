from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


__all__ = ['colorize','plot_1D','plot_amplitude','plot_phase',
           'plot_colorized']


def colorize(z):
    """ Returns RGB values for a complex color plot given a complex array
    This function returns a set of RGB values that can be used directly
    in a call to imshow based on an input complex numpy array (not a
    torch tensor representing a complex field)

    Args:
        z (array_like) : A complex-valued array
    Returns:
        list : A list of arrays for R,G, and B channels of an image.

    """

    amp = np.abs(z)
    rmin = 0
    rmax = np.max(amp)
    amp = np.where(amp < rmin, rmin, amp)
    amp = np.where(amp > rmax, rmax, amp)
    ph = np.angle(z, deg=1) + 90
    # HSV are values in range [0,1]
    h = (ph % 360) / 360
    s = 0.85 * np.ones_like(h)
    v = (amp - rmin) / (rmax - rmin)

    return hsv_to_rgb(np.dstack((h,s,v)))


def plot_1D(arr, fig = None, **kwargs):
    """Simple 1D plotter

    Args:
        im (numpy array) : A 1D array with dimensions (,N)
        fig (matplotlib.figure.Figure) : A matplotlib figure to use to plot. If None,
        a new figure is created with an Axes subplot at 111.
        **kwargs: Can be used to set any keyword arguments for the matplotlib.axes.Axes class
        (see https://matplotlib.org/api/axes_api.html#the-axes-class)
    """
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, **kwargs)
    plt.scatter(np.arange(arr.shape[-1]), arr)

    
def plot_amplitude(im, fig = None, basis = np.array([[0,-1], [-1,0], [0,0]]), **kwargs):
    """ Plots the amplitude of a complex Tensor or numpy array with dimensions NxMx2.
    Args:
        im (t.Tensor) : An image with dimensions NxMx2.
        fig (matplotlib.figure.Figure) : A matplotlib figure to use to plot. If None,
        a new figure is created with an Axes subplot at 111.
        basis (numpy array) : The probe basis, used to put the axis labels in real space units.
        Should have dimensions 3x2
        **kwargs: Can be used to set any keyword arguments for the matplotlib.axes.Axes class
        (see https://matplotlib.org/api/axes_api.html#the-axes-class)
    """
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, **kwargs)
    basis_norm = np.linalg.norm(basis, axis = 0)
    if isinstance(im, t.Tensor):
        absolute = cmath.cabs(im).detach().cpu().numpy()
    else:
        absolute = np.absolute(im)
    plt.imshow(absolute, cmap = 'viridis', extent = [0, absolute.shape[-1]*basis_norm[1], 0, absolute.shape[-2]*basis_norm[0]])
    plt.colorbar()
    return fig


def plot_phase(im, fig = None, basis =  np.array([[0,-1], [-1,0], [0,0]]), **kwargs):
    """ Plots the phase of a complex Tensor or numpy array with dimensions NxMx2.
    Args:
        im (t.Tensor) : An image with dimensions NxMx2.
        fig (matplotlib.figure.Figure) : A matplotlib figure to use to plot. If None,
        a new figure is created with an Axes subplot at 111.
        basis (numpy array) : The probe basis, used to put the axis labels in real space units.
        Should have dimensions 3x2
        **kwargs: Can be used to set any keyword arguments for the matplotlib.axes.Axes class
        (see https://matplotlib.org/api/axes_api.html#the-axes-class)
    """
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, **kwargs)
    # If the user has matplotlib >=3.0, use the preferred colormap
    if isinstance(im, t.Tensor):
        phase = cmath.cphase(im).detach().cpu().numpy()
    else:
        phase = np.angle(im)
    basis_norm = np.linalg.norm(basis, axis = 0)
    try: plt.imshow(phase, cmap = 'twilight', extent = [0, phase.shape[-1]*basis_norm[1], 0, phase.shape[-2]*basis_norm[0]])
    except: plt.imshow(phase, cmap = 'hsv', extent = [0, phase.shape[-1]*basis_norm[1], 0, phase.shape[-2]*basis_norm[0]])
    plt.colorbar()
    return fig


def plot_colorized(im, fig = None, basis =  np.array([[0,-1], [-1,0], [0,0]]), **kwargs):
    """ Plots the colorized version of a complex Tensor or numpy array with dimensions NxMx2.
    The darkness corresponds to the intensity of the image, and the color corresponds
    to the phase.

    Args:
        im (t.Tensor) : An image with dimensions NxMx2.
        fig (matplotlib.figure.Figure) : A matplotlib figure to use to plot. If None,
        a new figure is created with an Axes subplot at 111.
        basis (numpy array) : The probe basis, used to put the axis labels in real space units.
        Should have dimensions 3x2
        **kwargs: Can be used to set any keyword arguments for the matplotlib.axes.Axes class
        (see https://matplotlib.org/api/axes_api.html#the-axes-class)
    """
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, **kwargs)
    if isinstance(im, t.Tensor):
        im = cmath.torch_to_complex(im.detach().cpu())
    basis_norm = np.linalg.norm(basis, axis = 0)
    colorized = colorize(im)
    plt.imshow(colorized, extent = [0, im.shape[-1]*basis_norm[1], 0, im.shape[-2]*basis_norm[0]])
    return fig

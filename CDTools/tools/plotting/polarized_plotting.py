import torch as plot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import Slider
from matplotlib import ticker, patheffects

all = ['plot_components_amplitudes', 'plot_phase_ret',
       'plot_global_phase',
       'plot_fast_axes', 'plot_object_ellipses',
       'plot_probe_ellipse']

def iterator(a, b):
    '''
    A helper function we can use to facilitate to process of iterating over 2D arrays
    '''
    x, y = t.arange(x), t.arange(y)
    x, y = t.meshgrid(x, y)
    x, y = t.ravel(x), t.ravel(y)
    return (x, y)

def plot_probe_ellipse(a=1, b=1, phase_ret=0, scale=1, x0=4, y0=5):
    '''
    Given a probe vector at a point (x0, y0), visualizes ellipticity
    of its polarization

    Parameters:
    -----------
    a : 1D np.array
        An amplitude of the horizontal component of the probe vector
    b : 1D np.array
        An amplitude of the vertical component of the probe vector
    phase_ret : 1D np.array
        A phase difference in radians bettween the phases of the y and x components
    scale : int
        A scaling factor
    x0, y0: 1D np.array or float
        Defines the location of the vector to be plotted

    Returns:
    --------
    x, y set of points to plot a single ellipse
    '''
    theta = np.linspace(0, 2*np.pi, 20)
    x = x0 + scale * a * np.real(np.exp(1j * theta))
    y = y0 + scale * b * np.real(np.exp(1j * (theta + phase)))
    return x, y

def plot_attenuations(atten_slow=1, atten_fast=1, fast_ax_angle=0, scale=1, x0=4, y0=4):
    '''
    Plots attenuations along the fast and slow axes

    Parameters:
    -----------
    atten_slow: 1D np.array
        Attenuation along the slow axis
    atten_fast: 1D np.array
        Attenuation along the fast axis
    fast_ax_angle: 1D np.array
        An angle between the fast horizontal and fast axes
    phase_ret: 1D np.array
        A difference in phases gained by the slow and the fast components
        The most clockwise axis is always considered to be the fast one
    scale: 1D np.array
        A scaling factor
    x0, y0: 1D np.array or float
        Defines the location to be plotted at

    Returns:
    --------
    x, y set of points to plot a single Jones matrix of the object
    '''
    theta = np.linspace(0, 2*np.pi, 20)
    # collection of points to plot a fast axis
    x = x0 + scale * atten_fast * np.real(np.exp(1j * theta))
    x_f = x0 + (x - x0) * np.cos(angle)
    y_f = y0 + (x - x0) * np.sin(angle)
    # collection of point to plot a slow axis
    y = y_0 + scale * atten_slow * np.real(np.exp(1j * theta))
    x_s = x0 - (y - y0) * np.sin()
    y_s = y0 + (y - y0) * np.cos(angle)
    return x_f, y_f, x_s, y_s

def plot_fast_axis(fast_ax_angle=0, scale=1, x0=4, y0=4):
    '''
    Plots directions of the fast axes only
    '''
    xf, yf, xs, ys = plot_attenuations(fast_ax_angle=fast_ax_angle, atten_fast=1, atten_slow=0, scale=scale)
    return xf, yf

def plot_figures(shape, num_of_el_along_x=20, num_of_el_along_y=20,
                 phases=None, fast_ax_angles=None,
                 atten_fast=None, atten_slow=None, scale=5,
                 probe=False, attenuations=False, fast_axes=False):
    """
    All the parameters - np.arrays of shape (shape)
    """
    x_centers = np.linspace(1, shape[0] - 1, num_of_el_along_x)
    y_centers = np.linspace(1, shape[1] - 1, num_of_el_along_y)
    X, Y = np.meshgrid(x_centers, y_centers)
    X, Y = np.ravel(X), np.ravel(Y)
    xs, ys = np.array([]), np.array([])
    for x, y in zip(X, Y):
        k, m = int(x), int(y)
        if probe:
            xx, yy = plot_probe_ellipse(a=atten_fast[k, m], b=atten_slow[k, m],
                                        phase_ret=phases[k, m], scale=scale, x0=x, y0=y)
        elif attenuations:
            xf, yf, xs, ys = plot_attenuations(atten_slow=atten_slow[k, m], atten_fast=atten_fast[k, m],
                                       fast_ax_angle=fast_ax_angles[k, m], scale=scale, x0=x, y0=y)
        elif fast_axes:
            xx, yy = plot_fast_axis(fast_ax_angle=fast_ax_angles[k, m], scale=scale, x0=x, y0=y)

        if attenuations:
            plt.plot(xf, yf, c='b')
            plt.plot(xs, ys, c='b')
            plt.axis('equal')
        else:
            plt.plot(xx, yy, c='b')
            plt.axis('equal')
            

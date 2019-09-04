"""This module contains functions for plotting various important metrics

All the plotting functions here can accept torch input or numpy input,
to facilitate their use both for live inspection of running reconstructions
and for after-the-fact analysis. Utilities for plotting complex valued
images exist, as well as plotting scan patterns and nanomaps
"""

from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


__all__ = ['colorize', 'plot_amplitude', 'plot_phase',
           'plot_colorized', 'plot_translations', 'get_units_factor',
           'plot_nanomap']


def colorize(z):
    """ Returns RGB values for a complex color plot given a complex array
    This function returns a set of RGB values that can be used directly
    in a call to imshow based on an input complex numpy array (not a
    torch tensor representing a complex field)

    Parameters
    ----------
    z : array
        A complex-valued array
    Returns
    -------
    rgb : list(array) 
        A list of arrays for the R,G, and B channels of an image
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


def get_units_factor(units):
    """Gets the multiplicative factor associated with a length unit

    Parameters
    ----------
    units : str
        The abbreviation for the unit type
    
    Returns
    -------
    factor : float
        The factor meters / (unit)
    """
    
    u = units.lower()
    if u=='m':
        factor=1
    if u=='cm':
        factor=1e2
    if u=='mm':
        factor=1e3
    if u=='um':
        factor=1e6
    if u=='nm':
        factor=1e9
    if u=='a':
        factor=1e10
    if u=='pm':
        factor=1e12
    return factor

    
def plot_amplitude(im, fig = None, basis=None, units='um', cmap='viridis', **kwargs):
    """Plots the amplitude of a complex array with dimensions NxM
    
    If a figure is given explicitly, it will clear that existing figure and
    plot over it. Otherwise, it will generate a new figure.

    If a basis is explicitly passed, the image will be plotted in real-space
    coordinates
    
    Parameters
    ----------
    im : array
        An complex array with dimensions NxM
    fig : matplotlib.figure.Figure
        Default is a new figure, a matplotlib figure to use to plot
    basis : np.array
        Optional, the 3x2 probe basis
    units : str
        The length units to mark on the plot, default is um
    cmap : str
        Default is 'viridis', the colormap to plot with
    \\**kwargs
        All other args are passed to fig.add_subplot(111, \\**kwargs)

    Returns
    -------
    used_fig : matplotlib.figure.Figure
        The figure object that was actually plotted to.
    """
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, **kwargs)
    else:
        plt.figure(fig.number)
        plt.gcf().clear()

    if isinstance(im, t.Tensor):
        absolute = cmath.cabs(im).detach().cpu().numpy()
    else:
        absolute = np.absolute(im)

    #Plot in a basis if it exists, otherwise dont
    if basis is not None:
        if isinstance(basis,t.Tensor):
            basis = basis.detach().cpu().numpy()
        basis_norm = np.linalg.norm(basis, axis = 0)
        basis_norm = basis_norm * get_units_factor(units)
        
        extent = [0, absolute.shape[-1]*basis_norm[1], 0, absolute.shape[-2]*basis_norm[0]]
    else:
        extent=None
        
    plt.imshow(absolute, cmap = cmap, extent = extent)
    cbar = plt.colorbar()
    cbar.set_label('Amplitude (a.u.)')

    if basis is not None:
        plt.xlabel('X (' + units + ')')
        plt.ylabel('Y (' + units + ')')
    else:
        plt.xlabel('j (pixels)')
        plt.ylabel('i (pixels)')
        
    return fig


def plot_phase(im, fig=None, basis=None, units='um', cmap='auto', **kwargs):
    """ Plots the phase of a complex array with dimensions NxMx2

    If a figure is given explicitly, it will clear that existing figure and
    plot over it. Otherwise, it will generate a new figure.

    If a basis is explicitly passed, the image will be plotted in real-space
    coordinates
    
    Parameters
    ----------
    im : array
        An complex array with dimensions NxM
    fig : matplotlib.figure.Figure
        Default is a new figure, a matplotlib figure to use to plot
    basis : np.array
        Optional, the 3x2 probe basis
    units : str
        The length units to mark on the plot, default is um
    cmap : str
        Default is 'viridis', the colormap to plot with
    \\**kwargs
        All other args are passed to fig.add_subplot(111, \\**kwargs)

    Returns
    -------
    used_fig : matplotlib.figure.Figure
        The figure object that was actually plotted to.
    """
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, **kwargs)
    else:
        plt.figure(fig.number)
        plt.gcf().clear()

    if isinstance(im, t.Tensor):
        phase = cmath.cphase(im).detach().cpu().numpy()
    else:
        phase = np.angle(im)

    if basis is not None:
        if isinstance(basis,t.Tensor):
            basis = basis.detach().cpu().numpy()
        basis_norm = np.linalg.norm(basis, axis = 0)
        basis_norm = basis_norm * get_units_factor(units)
        
        extent = [0, phase.shape[-1]*basis_norm[1], 0, phase.shape[-2]*basis_norm[0]]
    else:
        extent=None

    
    # If the user has matplotlib >=3.0, use the preferred colormap
    if cmap == 'auto':
        try:
            plt.imshow(phase, cmap = 'twilight', extent=extent)
        except:
            plt.imshow(phase, cmap = 'hsv', extent=extent)
    else:
        plt.imshow(phase)#, cmap = cmap, extent=extent)
        
    cbar = plt.colorbar()
    cbar.set_label('Phase (rad)')
    
    if basis is not None:
        plt.xlabel('X (' + units + ')')
        plt.ylabel('Y (' + units + ')')
    else:
        plt.xlabel('j (pixels)')
        plt.ylabel('i (pixels)')
        
    return fig


def plot_colorized(im, fig=None, basis=None, units='um', **kwargs):
    """ Plots the colorized version of a complex array with dimensions NxM

    The darkness corresponds to the intensity of the image, and the color
    corresponds to the phase.

    If a figure is given explicitly, it will clear that existing figure and
    plot over it. Otherwise, it will generate a new figure.

    If a basis is explicitly passed, the image will be plotted in real-space
    coordinates
    
    Parameters
    ----------
    im : array
        An complex array with dimensions NxM
    fig : matplotlib.figure.Figure
        Default is a new figure, a matplotlib figure to use to plot
    basis : np.array
        Optional, the 3x2 probe basis
    units : str
        The length units to mark on the plot, default is um
    \\**kwargs
        All other args are passed to fig.add_subplot(111, \\**kwargs)

    Returns
    -------
    used_fig : matplotlib.figure.Figure
        The figure object that was actually plotted to.
    """
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, **kwargs)
    else:
        plt.figure(fig.number)
        plt.gcf().clear()
        
    if isinstance(im, t.Tensor):
        im = cmath.torch_to_complex(im.detach().cpu())

    if basis is not None:
        if isinstance(basis,t.Tensor):
            basis = basis.detach().cpu().numpy()
        basis_norm = np.linalg.norm(basis, axis = 0)
        basis_norm = basis_norm * get_units_factor(units)
        
        extent = [0, im.shape[-1]*basis_norm[1], 0, im.shape[-2]*basis_norm[0]]
    else:
        extent=None

    colorized = colorize(im)
    plt.imshow(colorized, extent=extent)

    if basis is not None:
        plt.xlabel('X (' + units + ')')
        plt.ylabel('Y (' + units + ')')
    else:
        plt.xlabel('j (pixels)')
        plt.ylabel('i (pixels)')
        
    return fig



def plot_translations(translations, fig=None, units='um', lines=True, **kwargs):
    """Plots a set of probe translations in a nicely formatted way
    
    Parameters
    ----------
    translations : array
        An Nx2 or Nx3 set of translations in real space
    fig : matplotlib.figure.Figure
        Default is a new figure, a matplotlib figure to use to plot
    units : str
        Default is um, units to report in (assuming input in m)
    lines : bool
        Whether to plot lines indicating the path taken
    \\**kwargs
        All other args are passed to fig.add_subplot(111, \\**kwargs)


    Returns
    -------
    used_fig : matplotlib.figure.Figure
        The figure object that was actually plotted to.
    """
    
    factor = get_units_factor(units)
    
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, **kwargs)
    else:
        plt.figure(fig.number)
        plt.gcf().clear()

    if isinstance(translations, t.Tensor):
        translations = translations.detach().cpu().numpy()
        
    translations = translations * factor
    plt.plot(translations[:,0], translations[:,1],'k.')
    if lines:
        plt.plot(translations[:,0], translations[:,1],'b-', linewidth=0.5)
    plt.xlabel('X (' + units + ')')
    plt.ylabel('Y (' + units + ')')

    return fig

    
def plot_nanomap(translations, values, fig=None, units='um', convention='probe'):
    """Plots a set of nanomap data in a flexible way
    
    Parameters
    ----------
    translations : array
        An Nx2 or Nx3 set of translations in real space
    values : array
        A length-N object of values associated with the translations
    fig : matplotlib.figure.Figure
        Default is a new figure, a matplotlib figure to use to plot
    units : str
        Default is um, units to report in (assuming input in m)
    convention : str
        Default is 'probe', alternative is 'obj'. Whether the translations refer to the probe or object.

    Returns
    -------
    used_fig : matplotlib.figure.Figure
        The figure object that was actually plotted to.
    """

    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)
        plt.gcf().clear()

    factor = get_units_factor(units)

    bbox = fig.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    if isinstance(translations, t.Tensor):
        trans = translations.detach().cpu().numpy()
    else:
        trans = np.array(translations)

    if isinstance(values, t.Tensor):
        values = values.detach().cpu().numpy()
    else:
        values = np.array(values)

    if convention.lower() != 'probe':
        trans = trans * -1
        
    s = bbox.width * bbox.height / trans.shape[0] * 72**2 #72 is points per inch
    s /= 4 # A rough value to make the size work out
    
    plt.scatter(factor * trans[:,0],factor * trans[:,1],s=s,c=values)
    
    plt.gca().set_facecolor('k')
    plt.xlabel('Translation x (' + units + ')')
    plt.ylabel('Translation y (' + units + ')')
    plt.colorbar()

    return fig

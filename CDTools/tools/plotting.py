from __future__ import division, print_function, absolute_import

from CDTools.tools import cmath
import torch as t
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


__all__ = ['colorize','plot_1D','plot_amplitude','plot_phase',
           'plot_colorized', 'plot_translations','get_units_factor',
           'plot_nanomap']


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


def get_units_factor(units):
    """Gets the multiplicative factor associated with a length unit

    Args:
        units (str) : The abbreviation for the unit type
    
    Returns:
        (float) : The factor meters / (unit)
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
    else:
        plt.figure(fig.number)
        plt.gcf().clear()

    plt.scatter(np.arange(arr.shape[-1]), arr)

    
def plot_amplitude(im, fig = None, basis=None, units='um', cmap='viridis', **kwargs):
    """ Plots the amplitude of a complex Tensor or numpy array with dimensions NxMx2.
    Args:
        im (t.Tensor) : An image with dimensions NxMx2.
        fig (matplotlib.figure.Figure) : A matplotlib figure to use to plot. If None,
        a new figure is created with an Axes subplot at 111.
        basis (numpy array) : Optional, the 3x2 probe basis, used to put the axis labels in real space units.
        units (str) : The units to convert the basis to
        cmap (str) : Default is 'viridis', the colormap to plot with
        **kwargs: Can be used to set any keyword arguments for the matplotlib.axes.Axes class
        (see https://matplotlib.org/api/axes_api.html#the-axes-class)
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
    """ Plots the phase of a complex Tensor or numpy array with dimensions NxMx2.
    Args:
        im (t.Tensor) : An image with dimensions NxMx2.
        fig (matplotlib.figure.Figure) : A matplotlib figure to use to plot. If None,
        a new figure is created with an Axes subplot at 111.
        basis (numpy array) : Optional, the 3x2 probe basis, used to put the axis labels in real space units.
        cmap (str) : Default is 'auto', which chooses between twilight and hsv based on availability.
        **kwargs: Can be used to set any keyword arguments for the matplotlib.axes.Axes class
        (see https://matplotlib.org/api/axes_api.html#the-axes-class)
    """
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, **kwargs)
    else:
        plt.figure(fig.number)
        plt.gcf().clear()

    # If the user has matplotlib >=3.0, use the preferred colormap
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

    if cmap == 'auto':
        try:
            plt.imshow(phase, cmap = 'twilight', extent=extent)
        except:
            plt.imshow(phase, cmap = 'hsv', extent=extent)
    else:
        plt.imshow(phase, cmap = cmap, extent=extent)
        
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
    """ Plots the colorized version of a complex Tensor or numpy array with dimensions NxMx2.
    The darkness corresponds to the intensity of the image, and the color corresponds
    to the phase.

    Args:
        im (t.Tensor) : An image with dimensions NxMx2.
        fig (matplotlib.figure.Figure) : A matplotlib figure to use to plot. If None,
        a new figure is created with an Axes subplot at 111.
        basis (numpy array) : Optional, the 3x2 probe basis, used to put the axis labels in real space units.
        **kwargs: Can be used to set any keyword arguments for the matplotlib.axes.Axes class
        (see https://matplotlib.org/api/axes_api.html#the-axes-class)
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



def plot_translations(translations, fig=None, units='um', lines=True):
    """Plots a set of probe translations in a nicely formatted way
    
    Args:
        translations: An Nx2 or Nx3 set of translations in real space
        fig : Optional, a figure to plot into
        units : Default is um, units to report in (assuming input in m)
        lines : Whether to plot the lines indicating the path

    Returns:
        None
    """
    
    factor = get_units_factor(units)
    
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111)
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


    
def plot_nanomap(translations, values, fig=None, units='um', convention='probe'):
    """Plots a set of nanomap data in a flexible way
    
    Args:
        translations : An Nx2 or Nx3 set of translations in real space
        values : a length-N object of values associated with the translations
        fig : Optional, a figure to plot into
        units : Default is um, units to report in (assuming input in m)
        lines : Whether to plot the lines indicating the path
        convention : 'probe' if the translations refer to probe translations, 'obj' if they refer to object translations

    Returns:
        None
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
    
    plt.gca().invert_xaxis()
    plt.gca().set_facecolor('k')
    plt.xlabel('Translation x (' + units + ')')
    plt.ylabel('Translation y (' + units + ')')
    plt.colorbar()



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
           'plot_nanomap', 'plot_real', 'plot_imag']


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
    if u=='um' or u=="$\\mu$m":
        factor=1e6
    if u=='nm':
        factor=1e9
    if u=='a':
        factor=1e10
    if u=='pm':
        factor=1e12
    return factor


def plot_image(im, plot_func=lambda x: x, fig=None, basis=None, units='$\\mu$m', cmap='viridis', cmap_label=None, **kwargs):
    """Plots an image with a colorbar and on an appropriate spatial grid
    
    If a figure is given explicitly, it will clear that existing figure and
    plot over it. Otherwise, it will generate a new figure.

    If a basis is explicitly passed, the image will be plotted in real-space
    coordinates

    Finally, if a function is passed to the plot_func argument, this function
    will be called on each slice of data before it is plotted. This is used
    internally to enable the plot_real, plot_image, plot_phase, etc. functions.
    

    Parameters
    ----------
    im : array
        An complex array with dimensions NxM
    plot_func : callable
        A function which maps numpy arrays to the image to be plotted
    fig : matplotlib.figure.Figure
        Default is a new figure, a matplotlib figure to use to plot
    basis : np.array
        Optional, the 3x2 probe basis
    units : str
        The length units to mark on the plot, default is um
    cmap : str
        Default is 'viridis', the colormap to plot with
    cmap_label : str
        What to label the colorbar when plotting
    \\**kwargs
        All other args are passed to fig.add_subplot(111, \\**kwargs)

    Returns
    -------
    used_fig : matplotlib.figure.Figure
        The figure object that was actually plotted to.
    """
    
    # convert to numpy
    if isinstance(im, t.Tensor):
        # If final dimension is 2, assume it is a complex array. If not,
        # assume it represents a real array
        if im.shape[-1] == 2:
            im = cmath.torch_to_complex(im.detach().cpu())
        else:
            im = im.detach().cpu().numpy()

    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, **kwargs)

    # This nukes everything and updates either the appropriate image from the
    # stack of images, or the only image if only a single image has been
    # given
    def make_plot(idx):
        plt.figure(fig.number)
        title = plt.gca().get_title()
        fig.clear()

        
        # If im only has two dimensions, this reshape will add a leading
        # dimension, and update will be called on index 0. If it has 3 or more
        # dimensions, then all the leading dimensions will be compressed into
        # one long dimension which can be scrolled through.
        s = im.shape
        reshaped_im = im.reshape(-1,s[-2],s[-1])
        num_images = reshaped_im.shape[0]
        fig.plot_idx = idx % num_images
        
        to_plot = plot_func(reshaped_im[fig.plot_idx])
        
        #Plot in a basis if it exists, otherwise dont
        if basis is not None:
            if isinstance(basis,t.Tensor):
                np_basis = basis.detach().cpu().numpy()
            else:
                np_basis = basis
            # This fails if the basis is not rectangular
            basis_norm = np.linalg.norm(np_basis, axis = 0)
            basis_norm = basis_norm * get_units_factor(units)

            extent = [0, to_plot.shape[-1]*basis_norm[1], 0,
                      to_plot.shape[-2]*basis_norm[0]]
        else:
            extent=None

        plt.imshow(to_plot, cmap = cmap, extent = extent)
        cbar = plt.colorbar()
        if cmap_label is not None:
            cbar.set_label(cmap_label)

        if basis is not None:
            plt.xlabel('X (' + units + ')')
            plt.ylabel('Y (' + units + ')')
        else:
            plt.xlabel('j (pixels)')
            plt.ylabel('i (pixels)')

            
        plt.title(title)

        if len(im.shape) >= 3:
            plt.text(0.03, 0.03, str(fig.plot_idx), fontsize=14, transform=plt.gcf().transFigure)
        return fig

    if hasattr(fig, 'plot_idx'):
        result = make_plot(fig.plot_idx)
    else:
        result = make_plot(0)

    update = make_plot
        
    
    def on_action(event):
        if not hasattr(event, 'button'):
            event.button = None
        if not hasattr(event, 'key'):
            event.key = None
                
        if event.key == 'up' or event.button == 'up':
            update(fig.plot_idx - 1)
        elif event.key == 'down' or event.button == 'down':
            update(fig.plot_idx + 1)
        plt.draw()

    if len(im.shape) >=3:
        if not hasattr(fig,'my_callbacks'):
            fig.my_callbacks = []

        for cid in fig.my_callbacks:
            fig.canvas.mpl_disconnect(cid)
        fig.my_callbacks = []
        fig.my_callbacks.append(fig.canvas.mpl_connect('key_press_event',on_action))
        fig.my_callbacks.append(fig.canvas.mpl_connect('scroll_event',on_action))
    
    return result
    

def plot_real(im, fig = None, basis=None, units='$\\mu$m', cmap='viridis', cmap_label='Real Part (a.u.)', **kwargs):
    """Plots the real part of a complex array with dimensions NxM

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
    cmap_label : str
        What to label the colorbar when plotting
    \\**kwargs
        All other args are passed to fig.add_subplot(111, \\**kwargs)

    Returns
    -------
    used_fig : matplotlib.figure.Figure
        The figure object that was actually plotted to.
    """
    plot_func = lambda x: np.real(x)
    return plot_image(im, plot_func=plot_func, fig=fig, basis=basis,
                      units=units, cmap=cmap, cmap_label=cmap_label,
                      **kwargs)
    


def plot_imag(im, fig = None, basis=None, units='$\\mu$m', cmap='viridis', cmap_label='Imaginary Part (a.u.)', **kwargs):
    """Plots the imaginary part of a complex array with dimensions NxM

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
    cmap_label : str
        What to label the colorbar when plotting
    \\**kwargs
        All other args are passed to fig.add_subplot(111, \\**kwargs)

    Returns
    -------
    used_fig : matplotlib.figure.Figure
        The figure object that was actually plotted to.
    """
    plot_func = lambda x: np.imag(x)
    return plot_image(im, plot_func=plot_func, fig=fig, basis=basis,
                      units=units, cmap=cmap, cmap_label=cmap_label,
                      **kwargs)


def plot_amplitude(im, fig = None, basis=None, units='$\\mu$m', cmap='viridis', cmap_label='Amplitude (a.u.)', **kwargs):
    """Plots the amplitude of a complex array with dimensions NxM

    If a figure is given explicitly, it will clear that existing figure and
    plot over it. Otherwise, it will generate a new figure.

    If a basis is explicitly passed, the image will be plotted in real-space
    coordinates.
    
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
    cmap_label : str
        What to label the colorbar when plotting
    \\**kwargs
        All other args are passed to fig.add_subplot(111, \\**kwargs)

    Returns
    -------
    used_fig : matplotlib.figure.Figure
        The figure object that was actually plotted to.
    """
    plot_func = lambda x: np.absolute(x)
    return plot_image(im, plot_func=plot_func, fig=fig, basis=basis,
                      units=units, cmap=cmap, cmap_label=cmap_label,
                      **kwargs)


def plot_phase(im, fig=None, basis=None, units='$\\mu$m', cmap='auto', cmap_label='Phase (rad)', **kwargs):
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
    cmap_label : str
        What to label the colorbar when plotting
    \\**kwargs
        All other args are passed to fig.add_subplot(111, \\**kwargs)

    Returns
    -------
    used_fig : matplotlib.figure.Figure
        The figure object that was actually plotted to.
    """
    if cmap == 'auto':
        if 'twilight' in plt.colormaps():
            cmap = 'twilight'
        elif 'hsv' in plt.colormaps():
            cmap = 'hsv'
        else:
            raise AttributeError('Neither twilight or hsv colormap exists in this screwed up matplotlib install')

    plot_func = lambda x: np.angle(x)
    return plot_image(im, plot_func=plot_func, fig=fig, basis=basis,
                      units=units, cmap=cmap, cmap_label=cmap_label,
                      **kwargs)


def plot_amplitude_surfacenorm():
    pass

def plot_colorized(im, fig=None, basis=None, units='$\\mu$m', **kwargs):
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
    plot_func = lambda x: colorize(x)
    return plot_image(im, plot_func=plot_func, fig=fig, basis=basis,
                      units=units, **kwargs)


def plot_translations(translations, fig=None, units='$\\mu$m', lines=True, **kwargs):
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


def plot_nanomap(translations, values, fig=None, units='$\\mu$m', convention='probe'):
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

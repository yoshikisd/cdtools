"""This module contains functions for plotting various important metrics

All the plotting functions here can accept torch input or numpy input,
to facilitate their use both for live inspection of running reconstructions
and for after-the-fact analysis. Utilities for plotting complex valued
images exist, as well as plotting scan patterns and nanomaps
"""

import torch as t
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib.widgets import Slider
from matplotlib import ticker, patheffects


__all__ = ['colorize', 'plot_amplitude', 'plot_phase',
           'plot_colorized', 'plot_translations', 'get_units_factor',
           'plot_nanomap', 'plot_real', 'plot_imag',
           'plot_nanomap_with_images']


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
            im = im.detach().cpu().numpy()
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

            extent = [0, im.shape[-1]*basis_norm[1], 0,
                      im.shape[-2]*basis_norm[0]]
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


def plot_translations(translations, fig=None, units='$\\mu$m', lines=True, invert_xaxis=True, **kwargs):
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
    invert_xaxis : bool
        Default is True. This flips the x axis to match the convention from .cxi files of viewing the image from the beam's perspective
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
    if invert_xaxis:
        plt.gca().invert_xaxis()
        
    if lines:
        plt.plot(translations[:,0], translations[:,1],'b-', linewidth=0.5)
    plt.xlabel('X (' + units + ')')
    plt.ylabel('Y (' + units + ')')

    return fig


def plot_nanomap(translations, values, fig=None, units='$\\mu$m', convention='probe', invert_xaxis=True):
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
    invert_xaxis : bool
        Default is True. This flips the x axis to match the convention from .cxi files of viewing the image from the beam's perspective

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
    if invert_xaxis:
        plt.gca().invert_xaxis()
    
    plt.gca().set_facecolor('k')
    plt.xlabel('Translation x (' + units + ')')
    plt.ylabel('Translation y (' + units + ')')
    plt.colorbar()

    return fig


def plot_nanomap_with_images(translations, get_image_func, values=None, mask=None, basis=None, fig=None, nanomap_units='$\\mu$m', image_units='$\\mu$m', convention='probe', image_title='Image', image_colorbar_title='Image Amplitude', nanomap_colorbar_title='Integrated Intensity', cmap='viridis', **kwargs):
    """Plots a nanomap, with an image or stack of images for each point

    In many situations, ptychography data or the output of ptychography
    reconstructions is formatted as a set of images associated with various
    points in real space. This function is designed to allow for browsing
    through this kind of data, by making it possible to visualize a

    """

    # This should pull heavily from the dataset.inspect function
    # In fact, I should be able to replace most of that function with a
    # call to this function once it's built
    # We start by making the figure and axes

    # The key will be writing this so it works okay when called in "update"
    # mode, i.e. on a figure that already has this thing showing.

    if fig is None:
        fig = plt.figure(figsize=(8,5.3))
    else:
        plt.figure(fig.number)
        plt.gcf().clear()
        if hasattr(fig, 'nanomap_cids'):
            for cid in fig.nanomap_cids:
                fig.canvas.mpl_disconnect(cid)

    # Does figsize work with the fig.subplots, or just for plt.subplots?
    axes = fig.subplots(1,2)

    fig.tight_layout(rect=[0.04, 0.09, 0.98, 0.96])
    plt.subplots_adjust(wspace=0.25) #avoids overlap of labels with plots
    axslider = plt.axes([0.15,0.06,0.75,0.03])

    # This gets the set of sizes for the points in the nanomap
    def calculate_sizes(idx):
        bbox = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        s0 = bbox.width * bbox.height / translations.shape[0] * 72**2 #72 is points per inch
        s0 /= 4 # A rough value to make the size work out
        s = np.ones(translations.shape[0]) * s0

        s[idx] *= 4
        return s


    def update_colorbar(im):
        #
        # This solves the problem of the colorbar being changed
        # when the forward and back buttons are used!!!
        #
        if hasattr(im, 'norecurse') and im.norecurse:
            im.norecurse=False
            return

        im.norecurse=True
        # This is needed to update the colorbar
        # only change limits if array contains multiple values
        if np.min(im.get_array()) != np.max(im.get_array()):
            im.set_clim(vmin=np.min(im.get_array()),
                        vmax=np.max(im.get_array()))

    #
    # The meatiest part of this program, here we just go through and
    # set up the plot how we want it
    #

    # First we set up the left-hand plot, which shows an overview map
    axes[0].set_title('Relative Displacement Map')

    translations = translations.detach().cpu().numpy()

    if convention.lower() != 'probe':
        translations = translations * -1

    s = calculate_sizes(0)

    nanomap_units_factor = get_units_factor(nanomap_units)
    nanomap = axes[0].scatter(nanomap_units_factor * translations[:,0],
                              nanomap_units_factor * translations[:,1],
                              s=s,c=values, picker=True)

    axes[0].invert_xaxis()
    axes[0].set_facecolor('k')
    axes[0].set_xlabel('Translation x ('+nanomap_units+')', labelpad=1)
    axes[0].set_ylabel('Translation y ('+nanomap_units+')', labelpad=1)
    cb1 = plt.colorbar(nanomap, ax=axes[0], orientation='horizontal',
                       format='%.2e',
                       ticks=ticker.LinearLocator(numticks=5),
                       pad=0.17,fraction=0.1)
    cb1.ax.set_title(nanomap_colorbar_title, size="medium", pad=5)
    cb1.ax.tick_params(labelrotation=20)
    if values is None:
        # This seems to do a good job of leaving the appropriate space
        # where the colorbar should have been to avoid stretching the
        # nanomap plot, while still not showing the (now useless) colorbar.
        cb1.remove()

    # Now we set up the second plot, which shows the individual
    # diffraction patterns
    axes[1].set_title(image_title)
    #Plot in a basis if it exists, otherwise dont
    if basis is not None:
        axes[1].set_xlabel('X (' + image_units + ')')
        axes[1].set_ylabel('Y (' + image_units + ')')

        example_im = get_image_func(0)
        if isinstance(example_im, t.Tensor):
            example_im = example_im.cpu().numpy()

        if isinstance(basis,t.Tensor):
            np_basis = basis.detach().cpu().numpy()
        else:
            np_basis = basis
        # This fails if the basis is not rectangular
        basis_norm = np.linalg.norm(np_basis, axis = 0)
        basis_norm = basis_norm * get_units_factor(image_units)

        extent = [0, example_im.shape[-1]*basis_norm[1], 0,
                  example_im.shape[-2]*basis_norm[0]]
    else:
        axes[1].set_xlabel('j (pixels)')
        axes[1].set_ylabel('i (pixels)')
        extent=None

    im=get_image_func(0)
    if len(im.shape) >= 3:
        im_idx=0
        axes[1].image_idx = im_idx
        im = im.reshape(-1,im.shape[-2],im.shape[-1])[im_idx]
        axes[1].text_box = axes[1].text(0.98, 0.98, str(im_idx), color='w',
                                        fontsize=14,
                                        horizontalalignment='right',
                                        verticalalignment='top',
                                        transform=axes[1].transAxes)
        axes[1].text_box.set_path_effects(
            [patheffects.Stroke(linewidth=2, foreground='black'),
             patheffects.Normal()])

    meas = axes[1].imshow(im, extent=extent, cmap=cmap)

    cb2 = plt.colorbar(meas, ax=axes[1], orientation='horizontal',
                       format='%.2e',
                       ticks=ticker.LinearLocator(numticks=5),
                       pad=0.17,fraction=0.1)
    cb2.ax.tick_params(labelrotation=20)
    cb2.ax.set_title(image_colorbar_title, size="medium", pad=5)
    cb2.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(meas))

    # This function handles all the updating, except for moving the
    # slider value. This is done because the slider widget is
    # ultimately responsible for triggering an update, so all other
    # updates are done by changing the slider widget value which
    # then triggers this
    def update(idx, im_idx=None):
        # We have to explicitly make it an integer because the slider will
        # output floats (even if they are still integer-valued)
        idx = int(idx)

        # Get the new data for this index
        im = get_image_func(idx)
        if len(im.shape) >= 3:
            if im_idx == None and hasattr(axes[1],'image_idx'):
                im_idx = axes[1].image_idx
            elif im_idx == None:
                im_idx=0
            axes[1].image_idx = im_idx
            axes[1].text_box.set_text(str(im_idx))
            im = im.reshape(-1,im.shape[-2],im.shape[-1])[im_idx]

        # Now we resize the nanomap to show the new selection
        axes[0].collections[0].set_sizes(calculate_sizes(idx))

        # And we update the data in the image as well

        ax_im = axes[1].images[-1]
        ax_im.set_data(im)
        update_colorbar(ax_im)


    #
    # Now we define the functions to handle various kinds of events
    # that can be thrown our way
    #

    # We start by creating the slider here, so it can be used
    # by the update hooks.
    slider = Slider(axslider, 'Image #', 0, translations.shape[0]-1, valstep=1, valfmt="%d")

    # This handles scroll wheel and keypress events
    def on_action(event):
        im = get_image_func(0)
        if event.inaxes is axes[1] and len(im.shape) >=3:
            # This is triggered if the data to display has more than 2
            # dimensions (i.e. is an image stack) and the event originates
            # while the mouse is within the image display
            im = im.reshape(-1,im.shape[-2],im.shape[-1])
            im_idx = axes[1].image_idx

            if event.key == 'up' or event.button == 'up' \
               or event.key == 'left':
                im_idx = (im_idx - 1) % im.shape[0]
            if event.key == 'down' or event.button == 'down' \
               or event.key == 'right':
                im_idx = (im_idx + 1) % im.shape[0]

            axes[1].image_idx=im_idx
            slider.set_val(slider.val)#update(slider.val,im_idx=im_idx)
            return # This prevents the rest from also happening

        # Otherwise the if statements can throw errors when the
        # event type isn't right, this way they just don't trigger
        if not hasattr(event, 'button'):
            event.button = None
        if not hasattr(event, 'key'):
            event.key = None

        if event.key == 'up' or event.button == 'up' or event.key == 'left':
            idx = slider.val - 1
        elif event.key == 'down' or event.button == 'down' or event.key == 'right':
            idx = slider.val + 1
        else:
            # This prevents errors from being thrown on irrelevant key
            # or mouse input
            return

        # Handle the wraparound and trigger the update
        idx = int(idx) % translations.shape[0]
        slider.set_val(idx)

    # This handles "pick" events in the nanomap
    def on_pick(event):
        # If we don't filter on type of event, this will also capture,
        # for example, scroll events that happen over the nanomap
        if event.mouseevent.button == 1:
            slider.set_val(event.ind[0])


    # Here we connect the various update functions
    cid1 = fig.canvas.mpl_connect('pick_event',on_pick)
    cid2 = fig.canvas.mpl_connect('key_press_event',on_action)
    cid3 = fig.canvas.mpl_connect('scroll_event',on_action)
    # It's so dumb that matplotlib doesn't automatically track this for you
    fig.nanomap_cids = [cid1,cid2,cid3]
    slider.on_changed(update)

    # Throw an extra update into the mix just to get rid of any things
    # (like the nanomap dot sizes) that otherwise would change on the
    # first update
    update(0)

    return fig

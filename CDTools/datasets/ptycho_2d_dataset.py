from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from copy import copy
import h5py
try:
    import pathlib
except ImportError:
    import pathlib2 as pathlib

from CDTools.datasets import CDataset
from CDTools.tools import data as cdtdata
from CDTools.tools import plotting
from torch.utils import data as torchdata
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import ticker

__all__ = ['Ptycho2DDataset']


class Ptycho2DDataset(CDataset):
    """The standard dataset for a 2D ptychography scan

    Subclasses datasets.CDataset

    This class loads and saves 2D ptychography scan data from .cxi files.
    It should save and load files compatible with most reconstruction
    programs, although it is only tested against SHARP.
    """
    def __init__(self, translations, patterns, axes=None, *args, **kwargs):
        """The __init__ function allows construction from python objects.


        The detector_geometry dictionary is defined to have the
        entries defined by the outputs of data.get_detector_geometry.


        Parameters
        ----------
        translations : array
            An nx3 array containing the probe translations at each scan point
        patterns : array
            An nxmxl array containing the full stack of measured diffraction patterns
        axes : list(str)
            A list of names for the axes of the probe translations
        entry_info : dict
            A dictionary containing the entry_info metadata
        sample_info : dict
            A dictionary containing the sample_info metadata
        wavelength : float
            The wavelength of light used in the experiment
        detector_geometry : dict
            A dictionary containing the various detector geometry
            parameters
        mask : array
            A mask for the detector, defined as 1 for live pixels, 0
            for dead
        background : array
            An initial guess for the not-previously-subtracted
            detector background
        """


        super(Ptycho2DDataset,self).__init__(*args, **kwargs)
        self.axes = copy(axes)
        self.translations = t.Tensor(translations).clone()
        self.patterns = t.Tensor(patterns).clone()
        if self.mask is None:
            self.mask = t.ones(self.patterns.shape[-2:]).to(dtype=t.bool)
        self.mask.masked_fill_(t.isnan(t.sum(self.patterns,dim=(0,))),0)
        self.patterns.masked_fill_(t.isnan(self.patterns),0)


    def __len__(self):
        return self.patterns.shape[0]

    def _load(self, index):
        """ Internal function to load data

        This function is used internally by the global __getitem__ function
        defined in the base class, which handles moving data around when
        the dataset is (for example) storing the data on the CPU but
        getting data as GPU tensors.

        It loads data in the format (inputs, output)
        
        The inputs for a 2D ptychogaphy data set are:

        1) The indices of the patterns to use
        2) The recorded probe positions associated with those points

        Parameters
        ----------
        index : int or slice
             The index or indices of the scan points to use

        Returns
        -------
        inputs : tuple
             A tuple of the inputs to the related forward models
        outputs : tuple
             The output pattern or stack of output patterns
        """
        return (index, self.translations[index]), self.patterns[index]


    def to(self, *args, **kwargs):
        """Sends the relevant data to the given device and dtype

        This function sends the stored translations, patterns,
        mask and background to the specified device and dtype

        Accepts the same parameters as torch.Tensor.to
        """
        super(Ptycho2DDataset,self).to(*args,**kwargs)
        self.translations = self.translations.to(*args, **kwargs)
        self.patterns = self.patterns.to(*args, **kwargs)


    # It sucks that I can't reuse the base factory method here,
    # perhaps there is a way but I couldn't figure it out.
    @classmethod
    def from_cxi(cls, cxi_file):
        """Generates a new CDataset from a .cxi file directly

        This generates a new Ptycho2DDataset from a .cxi file storing
        a 2D ptychography scan.

        Parameters
        ----------
        file : str, pathlib.Path, or h5py.File
            The .cxi file to load from

        Returns
        -------
        dataset : Ptycho2DDataset
            The constructed dataset object
        """
        # If a bare string is passed
        if isinstance(cxi_file, str) or isinstance(cxi_file, pathlib.Path):
            with h5py.File(cxi_file,'r') as f:
                return cls.from_cxi(f)

        # Generate a base dataset
        dataset = CDataset.from_cxi(cxi_file)
        # Mutate the class to this subclass (BasicPtychoDataset)
        dataset.__class__ = cls
        
        # Load the data that is only relevant for this class
        patterns, axes = cdtdata.get_data(cxi_file)
        translations = cdtdata.get_ptycho_translations(cxi_file)

        # And now re-do the stuff from __init__
        dataset.translations = t.Tensor(translations).clone()
        dataset.patterns = t.Tensor(patterns).clone()
        dataset.axes = axes
        if dataset.mask is None:
            dataset.mask = t.ones(dataset.patterns.shape[-2:]).to(dtype=t.bool)

        return dataset
    

    def to_cxi(self, cxi_file):
        """Saves out a Ptycho2DDataset as a .cxi file

        This function saves all the compatible information in a
        Ptycho2DDataset object into a .cxi file. This saved .cxi file
        should be compatible with any standard .cxi file based
        reconstruction tool, such as SHARP.

        Parameters
        ----------
        cxi_file : str, pathlib.Path, or h5py.File
            The .cxi file to write to
        """

        # If a bare string is passed
        if isinstance(cxi_file, str) or isinstance(cxi_file, pathlib.Path):
            with cdtdata.create_cxi(cxi_file) as f:
                return self.to_cxi(f)

        super(Ptycho2DDataset,self).to_cxi(cxi_file)
        if hasattr(self, 'axes'):
            cdtdata.add_data(cxi_file, self.patterns, axes=self.axes)
        else:
            cdtdata.add_data(cxi_file, self.patterns)
        cdtdata.add_ptycho_translations(cxi_file, self.translations)


    def inspect(self, logarithmic=True):
        """Launches an interactive plot for perusing the data

        This launches an interactive plotting tool in matplotlib that
        shows the spatial map constructed from the integrated intensity
        at each position on the left, next to a panel on the right that
        can display a base-10 log plot of the detector readout at each
        position.
        """
        
        # We start by making the figure and axes
        fig, axes = plt.subplots(1,2,figsize=(8,5.3))
        fig.tight_layout(rect=[0.04, 0.09, 0.98, 0.96])
        axslider = plt.axes([0.15,0.06,0.75,0.03])

        
        #
        # Then we define some helper functions for getting the right data
        # that are used both in the initial setup and the updates
        #
        
        def get_data(idx):
            inputs, output = self[idx]
            meas_data = output.detach().cpu().numpy()
            if hasattr(self, 'mask') and self.mask is not None:
                mask = self.mask.detach().cpu().numpy()
            else:
                mask = 1
            
            return mask, meas_data

        
        def calculate_sizes(idx):
            bbox = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            s0 = bbox.width * bbox.height / translations.shape[0] * 72**2 #72 is points per inch
            s0 /= 4 # A rough value to make the size work out
            s = np.ones(len(self)) * s0
            
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
            im.set_clim(vmin=np.min(im.get_array()),
                        vmax=np.max(im.get_array()))

        #
        # The meatiest part of this program, here we just go through and
        # set up the plot how we want it
        #

        # First we set up the left-hand plot, which shows an overview map
        axes[0].set_title('Relative Displacement Map')
        
        translations = self.translations.detach().cpu().numpy()
        nanomap_values = (self.mask.to(t.float32) * self.patterns).sum(dim=(1,2)).detach().cpu().numpy()
        s = calculate_sizes(0)
        
        nanomap = axes[0].scatter(1e6 * translations[:,0],1e6 * translations[:,1],s=s,c=nanomap_values, picker=True)
        
        axes[0].invert_xaxis()
        axes[0].set_facecolor('k')
        axes[0].set_xlabel('Translation x (um)', labelpad=1)
        axes[0].set_ylabel('Translation y (um)', labelpad=1)
        cb1 = plt.colorbar(nanomap, ax=axes[0], orientation='horizontal',
                           format='%.2e',
                           ticks=ticker.LinearLocator(numticks=5),
                           pad=0.17,fraction=0.1)
        cb1.ax.set_title('Integrated Intensity', size="medium", pad=5)
        cb1.ax.tick_params(labelrotation=20)


        # Now we set up the second plot, which shows the individual
        # diffraction patterns
        axes[1].set_title('Diffraction Pattern')
        mask, meas_data = get_data(0)
        if logarithmic:
            meas = axes[1].imshow(np.log(meas_data) / np.log(10) * mask)
        else:
            meas = axes[1].imshow(meas_data * mask)
            
        cb2 = plt.colorbar(meas, ax=axes[1], orientation='horizontal',
                           format='%.2e',
                           ticks=ticker.LinearLocator(numticks=5),
                           pad=0.17,fraction=0.1)
        cb2.ax.tick_params(labelrotation=20)
        cb2.ax.set_title('Pixel Intensity', size="medium", pad=5)
        cb2.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(meas))

        # This function handles all the updating, except for moving the
        # slider value. This is done because the slider widget is
        # ultimately responsible for triggering an update, so all other
        # updates are done by changing the slider widget value which
        # then triggers this
        def update(idx):
            # We have to explicitly make it an integer because the slider will
            # output floats (even if they are still integer-valued)
            idx = int(idx)

            # Get the new data for this index
            mask, meas_data = get_data(idx)

            # Now we resize the nanomap to show the new selection
            axes[0].collections[0].set_sizes(calculate_sizes(idx))

            # And we update the data in the image as well
            meas = axes[1].images[-1]
            if logarithmic:
                meas.set_data(np.log(meas_data) / np.log(10) * mask)
            else:
                meas.set_data(meas_data * mask)

            update_colorbar(meas)

   
        #
        # Now we define the functions to handle various kinds of events
        # that can be thrown our way
        #
        
        # We start by creating the slider here, so it can be used
        # by the update hooks.
        slider = Slider(axslider, 'Pattern #', 0, len(self)-1, valstep=1, valfmt="%d")

        # This handles scroll wheel and keypress events
        def on_action(event):
            # Otherwise the if statements can throw errors when the
            # event type isn't right, this way they just don't trigger
            if not hasattr(event, 'button'):
                event.button = None
            if not hasattr(event, 'key'):
                event.key = None

            if event.key == 'up' or event.button == 'up':
                idx = slider.val - 1
            elif event.key == 'down' or event.button == 'down':
                idx = slider.val + 1

            # Handle the wraparound and trigger the update
            idx = int(idx) % len(self)
            slider.set_val(idx)

        # This handles "pick" events in the nanomap
        def on_pick(event):
            # If we don't filter on type of event, this will also capture,
            # for example, scroll events that happen over the nanomap
            if event.mouseevent.button == 1:
                slider.set_val(event.ind[0])


        # Here we connect the various update functions
        fig.canvas.mpl_connect('pick_event',on_pick)
        fig.canvas.mpl_connect('key_press_event',on_action)
        fig.canvas.mpl_connect('scroll_event',on_action)
        slider.on_changed(update)

        # Throw an extra update into the mix just to get rid of any things
        # (like the nanomap dot sizes) that otherwise would change on the
        # first update
        update(0)
        

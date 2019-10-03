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
        self.translations = t.tensor(translations)
        self.patterns = t.tensor(patterns)
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

        entry_info = cdtdata.get_entry_info(cxi_file)
        sample_info = cdtdata.get_sample_info(cxi_file)
        wavelength = cdtdata.get_wavelength(cxi_file)
        distance, basis, corner = cdtdata.get_detector_geometry(cxi_file)
        detector_geometry = {'distance' : distance,
                             'basis'    : basis,
                             'corner'   : corner}
        mask = cdtdata.get_mask(cxi_file)

        dark = cdtdata.get_dark(cxi_file)
        patterns, axes = cdtdata.get_data(cxi_file)

        translations = cdtdata.get_ptycho_translations(cxi_file)
        return cls(translations, patterns, axes=axes,
                   entry_info = entry_info,
                   sample_info = sample_info,
                   wavelength=wavelength,
                   detector_geometry=detector_geometry,
                   mask=mask, background=dark)


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
            with h5py.File(cxi_file,'w') as f:
                return self.to_cxi(f)

        super(Ptycho2DDataset,self).to_cxi(cxi_file)
        cdtdata.add_data(cxi_file, self.patterns, axes=self.axes)
        cdtdata.add_ptycho_translations(cxi_file, self.translations)


    def inspect(self, logarithmic=True):
        """Launches an interactive plot for perusing the data

        This launches an interactive plotting tool in matplotlib that
        shows the spatial map constructed from the integrated intensity
        at each position on the left, next to a panel on the right that
        can display a base-10 log plot of the detector readout at each
        position.
        """
        fig, axes = plt.subplots(1,2,figsize=(8,5.3))
        fig.tight_layout(rect=[0.04, 0.09, 0.98, 0.96])
        axslider = plt.axes([0.15,0.06,0.75,0.03])

        translations = self.translations.detach().cpu().numpy()
        nanomap_values = (self.mask.to(t.float32) * self.patterns).sum(dim=(1,2)).detach().cpu().numpy()

        def update_colorbar(im):
            # If the update brought the colorbar out of whack
            # (say, from clicking back in the navbar)
            # Holy fuck this was annoying. Sorry future for how
            # crappy this solution is.
            if hasattr(im, 'norecurse') and im.norecurse:
                im.norecurse=False
                return

            im.norecurse=True
            im.colorbar.set_clim(vmin=np.min(im.get_array()),vmax=np.max(im.get_array()))
            im.colorbar.ax.set_ylim(0,1)
            im.colorbar.set_ticks(ticker.LinearLocator(numticks=5))
            im.colorbar.draw_all()

        def on_pick(event):
            update(event.ind[0])
            self.slider.set_val(fig.pattern_idx)
            plt.draw()

        def update(idx):
            idx = int(idx) % len(self)
            fig.pattern_idx = idx
            updating = True if len(axes[1].images) >= 1 else False

            inputs, output = self[idx]
            meas_data = output.detach().cpu().numpy()
            if hasattr(self, 'mask') and self.mask is not None:
                mask = self.mask.detach().cpu().numpy()
            else:
                mask = 1

            if not updating:
                axes[0].set_title('Relative Displacement Map')
                axes[1].set_title('Diffraction Pattern')

                bbox = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())

                s0 = bbox.width * bbox.height / translations.shape[0] * 72**2 #72 is points per inch
                s0 /= 4 # A rough value to make the size work out
                s = np.ones(len(nanomap_values)) * s0

                s[idx] *= 4

                nanomap = axes[0].scatter(1e6 * translations[:,0],1e6 * translations[:,1],s=s,c=nanomap_values, picker=True)
                fig.canvas.mpl_connect('pick_event',on_pick)

                axes[0].invert_xaxis()
                axes[0].set_facecolor('k')
                axes[0].set_xlabel('Translation x (um)', labelpad=1)
                axes[0].set_ylabel('Translation y (um)', labelpad=1)
                cb1 = plt.colorbar(nanomap, ax=axes[0], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.17,fraction=0.1)
                cb1.ax.set_title('Integrated Intensity', size="medium", pad=5)
                cb1.ax.tick_params(labelrotation=20)

                if logarithmic:
                    meas = axes[1].imshow(np.log(meas_data) / np.log(10) * mask)
                else:
                    meas = axes[1].imshow(meas_data * mask)

                cb2 = plt.colorbar(meas, ax=axes[1], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.17,fraction=0.1)
                cb2.ax.tick_params(labelrotation=20)
                cb2.ax.set_title('Pixel Intensity', size="medium", pad=5)
                cb2.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(meas))

            else:
                bbox = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())

                s0 = bbox.width * bbox.height / translations.shape[0] * 72**2 #72 is points per inch
                s0 /= 4 # A rough value to make the size work out
                s = np.ones(len(nanomap_values)) * s0
                s[idx] *= 4

                axes[0].clear()
                nanomap = axes[0].scatter(1e6 * translations[:,0],1e6 * translations[:,1],s=s,c=nanomap_values, picker=True)
                fig.canvas.mpl_connect('pick_event',on_pick)

                axes[0].set_title('Relative Displacement Map')
                axes[0].invert_xaxis()
                axes[0].set_facecolor('k')
                axes[0].set_xlabel('Translation x (um)')
                axes[0].set_ylabel('Translation y (um)')



                meas = axes[1].images[-1]
                if logarithmic:
                    meas.set_data(np.log(meas_data) / np.log(10) * mask)
                else:
                    meas.set_data(meas_data * mask)

                update_colorbar(meas)


        # This is dumb but the slider doesn't work unless a reference to it is
        # kept somewhere...
        self.slider = Slider(axslider, 'Pattern #', 0, len(self)-1, valstep=1, valfmt="%d")
        self.slider.on_changed(update)

        def on_action(event):
            if not hasattr(event, 'button'):
                event.button = None
            if not hasattr(event, 'key'):
                event.key = None

            if event.key == 'up' or event.button == 'up':
                update(fig.pattern_idx - 1)
            elif event.key == 'down' or event.button == 'down':
                update(fig.pattern_idx + 1)
            self.slider.set_val(fig.pattern_idx)
            plt.draw()

        fig.canvas.mpl_connect('key_press_event',on_action)
        fig.canvas.mpl_connect('scroll_event',on_action)
        update(0)

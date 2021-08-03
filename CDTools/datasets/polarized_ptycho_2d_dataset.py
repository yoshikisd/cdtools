from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from copy import copy
import h5py
import pathlib
from CDTools.datasets import CDataset, Ptycho2DDataset
from CDTools.tools import data as cdtdata
from CDTools.tools import plotting
from torch.utils import data as torchdata
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import ticker

__all__ = ['PolarizedPtycho2DDataset']


class PolarizedPtycho2DDataset(Ptycho2DDataset):
    """The standard dataset for a 2D ptychography scan

    Subclasses datasets.CDataset

    This class loads and saves 2D ptychography scan data from .cxi files.
    It should save and load files compatible with most reconstruction
    programs, although it is only tested against SHARP.
    """

    def __init__(self, translations, polarizer_angles, analyzer_angles, patterns, axes=None, *args, **kwargs):
        """The __init__ function allows construction from python objects.

        The detector_geometry dictionary is defined to have the
        entries defined by the outputs of data.get_detector_geometry.


        Parameters
        ----------
        translations : array (nx3 t.tensor when polarized=True)
            An nx3 array containing the probe translations at each scan point
        patterns : array
            An nxmxl array containing the full stack of measured diffraction patterns
        axes : list(str)
            A
             of names for the axes of the probe translations
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
        
        super(PolarizedPtycho2DDataset,self).__init__(translations, patterns,
                                             *args, **kwargs)
        
        
        # self.polarizer = t.tensor(polarizer_angles, dtype=t.float32)
        # self.analyzer = t.tensor(analyzer_angles, dtype=t.float32)
        

        polarizer = []
        analyzer = []
        for k in range(t.tensor(translations).shape[0]):
            polarizer.append((k//3)%3)
            analyzer.append((k%3))
        self.polarizer = t.tensor(polarizer)
        self.analyzer = t.tensor(analyzer)

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
        3) The angles of the polarizers if polarized=True

        Parameters
        ----------
        index (polarized=False): int or slice
             The index or indices of the scan points to use

        index (polarized=True): 
            tuple ((phi1, phi2), ind) 

            ind - index or indices of the scan points to use (in a (phi1, phi2) polarization state), int or slice
            (phi1, phi2) - angles of the 1st and 2nd polarizers, ints

        Returns
        -------
        inputs : tuple
             A tuple of the inputs to the related forward models
             if polarized: inputs = ((phi1, phi2), ind, transl[(phi1, phi2)][ind])
        outputs : tuple
             The output pattern or stack of output patterns
        """
        return ((index, self.translations[index],
                 self.polarizer[index], self.analyzer[index]),
                self.patterns[index])

    def to(self, *args, **kwargs):
        """Sends the relevant data to the given device and dtype

        This function sends the stored translations, patterns,
        mask and background to the specified device and dtype

        Accepts the same parameters as torch.Tensor.to
        """
        super(PolarizedPtycho2DDataset, self).to(*args, **kwargs)
        self.polarizer = self.polarizer.to(*args, **kwargs)
        self.analyzer = self.analyzer.to(*args, **kwargs)


    # It sucks that I can't reuse the base factory method here,
    # perhaps there is a way but I couldn't figure it out.
    @classmethod
    def from_cxi(cls, cxi_file):
        """Generates a new CDataset from a .cxi file directly

        This generates a new PolarizedPtycho2DDataset from a .cxi file storing
        a 2D ptychography scan.

        Parameters
        ----------
        file : str, pathlib.Path, or h5py.File
            The .cxi file to load from

        Returns
        -------
        dataset : PolarizedPtycho2DDataset
            The constructed dataset object
        """
        # If a bare string is passed
        if isinstance(cxi_file, str) or isinstance(cxi_file, pathlib.Path):
            with h5py.File(cxi_file, 'r') as f:
                return cls.from_cxi(f)

        # Generate a base dataset
        dataset = Ptycho2DDataset.from_cxi(cxi_file)

        # Mutate the class to this subclass (PolarizedPtycho2DDataset)
        dataset.__class__ = cls

        # Now, we save out the polarizer and analyzer states
        # polarizer = cdtdata.get_shot_to_shot_info(cxi_file, 'polarizer_angle')
        # analyzer = cdtdata.get_shot_to_shot_info(cxi_file, 'analyzer_angle')

        polarizer = []
        analyzer = []
        for k in range(dataset.translations.shape[0]):
            polarizer.append((k//3)%3)
            analyzer.append((k%3))
        dataset.polarizer = t.tensor(polarizer)
        dataset.analyzer = t.tensor(analyzer)


        dataset.analyzer = t.tensor(analyzer, dtype=t.float32)
        dataset.polarizer = t.tensor(polarizer, dtype=t.float32)
        
        return dataset

    def to_cxi(self, cxi_file, polarized=False):
        """Saves out a PolarizedPtycho2DDataset as a .cxi file

        This function saves all the compatible information in a
        PolarizedPtycho2DDataset object into a .cxi file. This saved .cxi file
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

        # This saves the translations, patterns, etc.
        super(PolarizedPtycho2DDataset, self).to_cxi(cxi_file)

        # Now, we save out the polarizer and analyzer states
        cdtdata.add_shot_to_shot_info(cxi_file, self.polarizer,
                                      'polarizer_angle')
        cdtdata.add_shot_to_shot_info(cxi_file, self.analyzer,
                                      'analyzer_angle')

    def inspect(self, logarithmic=True, units='um'):
        """Launches an interactive plot for perusing the data

        This launches an interactive plotting tool in matplotlib that
        shows the spatial map constructed from the integrated intensity
        at each position on the left, next to a panel on the right that
        can display a base-10 log plot of the detector readout at each
        position.
        """


        def get_images(idx):
            inputs, output = self[idx]
            meas_data = output.detach().cpu().numpy()
            if hasattr(self, 'mask') and self.mask is not None:
                mask = self.mask.detach().cpu().numpy()
            else:
                mask = 1
                
            if logarithmic:
                return np.log(meas_data) / np.log(10) * mask
            else:
                return meas_data * mask

        translations = self.translations.detach().cpu().numpy()
        nanomap_values = (self.mask.to(t.float32) * self.patterns).sum(dim=(1,2)).detach().cpu().numpy()

        if logarithmic:
            cbar_title='Log Base 10 of Diffraction Intensity'
        else:
            cbar_title='Diffraction Intensity'
        
        plotting.plot_nanomap_with_images(self.translations.detach().cpu(), get_images, values=nanomap_values, nanomap_units=units, image_title='Diffraction Pattern', image_colorbar_title=cbar_title)


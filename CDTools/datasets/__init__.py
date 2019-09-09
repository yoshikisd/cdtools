from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from copy import copy
import h5py
import pathlib

from CDTools.tools import data as cdtdata
from CDTools.tools import plotting
from torch.utils import data as torchdata
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import ticker


#
# This loads and stores all the kinds of metadata that are common to
# All different kinds of diffraction experiments
# Other datasets can subclass this and not worry about loading and
# saving that metadata.
#

class CDataset(torchdata.Dataset):

    def __init__(self, entry_info=None, sample_info=None,
                 wavelength=None,
                 detector_geometry=None, mask=None,
                 background=None):

        # Force pass-by-value-like behavior to stop strangeness
        self.entry_info = copy(entry_info)
        self.sample_info = copy(sample_info)
        self.wavelength = wavelength
        self.detector_geometry = copy(detector_geometry)
        if mask is not None:
            self.mask = t.tensor(mask)
        else:
            self.mask = None
        if background is not None:
            self.background = t.Tensor(background)
        else:
            self.background = None
    
        self.get_as(device='cpu')

            
    def to(self,*args,**kwargs):
        # The mask should always stay a uint8, but it should switch devices
        mask_kwargs = copy(kwargs)
        try:
            mask_kwargs.pop('dtype')
        except KeyError as r:
            pass
        
        if self.mask is not None:
            self.mask = self.mask.to(*args,**mask_kwargs)          
        if self.background is not None:
            self.background = self.background.to(*args,**kwargs)          


    def get_as(self, *args, **kwargs):
        self.get_as_args = (args, kwargs)


    def __getitem__(self, index):
        # Deals with loading to appropriate device/dtype, if
        # specified via a call to get_as
        inputs, outputs = self._load(index)
        if hasattr(self, 'get_as_args'):
            outputs = outputs.to(*self.get_as_args[0],**self.get_as_args[1])
            moved_inputs = []
            for inp in inputs:
                try:
                    moved_inputs.append(inp.to(*self.get_as_args[0],**self.get_as_args[1]) )
                except:
                    moved_inputs.append(inp)
        else:
            moved_inputs = inputs
        return moved_inputs, outputs

    
    def _load(self, index):
        # Internal function to load data
        raise NotImplementedError()
        
            
    @classmethod
    def from_cxi(cls, cxi_file):

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
        return cls(entry_info = entry_info,
                   sample_info = sample_info,
                   wavelength=wavelength,
                   detector_geometry=detector_geometry,
                   mask=mask, background=dark)
    
    
    def to_cxi(self, cxi_file):
        if self.entry_info is not None:
            cdtdata.add_entry_info(cxi_file, self.entry_info)
        if self.sample_info is not None:
            cdtdata.add_sample_info(cxi_file, self.sample_info)
        if self.wavelength is not None:
            cdtdata.add_source(cxi_file, self.wavelength)
        if self.detector_geometry is not None:
            if 'corner' in self.detector_geometry:
                corner = self.detector_geometry['corner']
            else:
                corner = None
            cdtdata.add_detector(cxi_file,
                               self.detector_geometry['distance'],
                               self.detector_geometry['basis'],
                               corner = corner)
        if self.mask is not None:
            cdtdata.add_mask(cxi_file, self.mask)
        if self.background is not None:
            cdtdata.add_dark(cxi_file, self.background)
        

from CDTools.datasets.ptycho_2d_dataset import Ptycho2DDataset

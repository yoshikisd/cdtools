from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from copy import copy

from CDTools.tools import data as cdtdata
from torch.utils import data as torchdata

__all__ = ['CDataset', 'Ptycho_2D_Dataset']


#
# This loads and stores all the kinds of metadata that are common to
# All different kinds of diffraction experiments
# Other datasets can subclass this and not worry about loading and
# saving that metadata.
#

class CDataset(torchdata.Dataset):

    def __init__(self, entry_info=None, sample_info=None,
                 wavelength=None,
                 detector_geometry=None, mask=None):

        # Force pass-by-value-like behavior to stop strangeness
        self.entry_info = copy(entry_info)
        self.sample_info = copy(sample_info)
        self.wavelength = wavelength
        self.detector_geometry = copy(detector_geometry)
        if mask is not None:
            self.mask = t.tensor(mask)
        else:
            self.mask = None
    
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
        entry_info = cdtdata.get_entry_info(cxi_file)
        sample_info = cdtdata.get_sample_info(cxi_file)
        wavelength = cdtdata.get_wavelength(cxi_file)
        distance, basis, corner = cdtdata.get_detector_geometry(cxi_file)
        detector_geometry = {'distance' : distance,
                             'basis'    : basis,
                             'corner'   : corner}
        mask = cdtdata.get_mask(cxi_file)
        return cls(entry_info = entry_info,
                   sample_info = sample_info,
                   wavelength=wavelength,
                   detector_geometry=detector_geometry,
                   mask=mask)
    
    
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
        


#
# This is the standard dataset for a 2D ptychography experiment,
# which saves and loads files compatible with most reconstruction
# programs (only tested against SHARP)
#

class Ptycho_2D_Dataset(CDataset):

    def __init__(self, translations, patterns, axes=None, *args, **kwargs):

        super(Ptycho_2D_Dataset,self).__init__(*args, **kwargs)
        self.axes = copy(axes)
        self.translations = t.tensor(translations)
        self.patterns = t.tensor(patterns)
        

    def __len__(self):
        return self.patterns.shape[0]

    def _load(self, index):
        return (index, self.translations[index]), self.patterns[index]


    def to(self, *args, **kwargs):
        super(Ptycho_2D_Dataset,self).to(*args,**kwargs)
        self.translations = self.translations.to(*args, **kwargs)
        self.patterns = self.patterns.to(*args, **kwargs)
        

    # It sucks that I can't reuse the base factory method here,
    # perhaps there is a way but I couldn't figure it out.
    @classmethod
    def from_cxi(cls, cxi_file):
        entry_info = cdtdata.get_entry_info(cxi_file)
        sample_info = cdtdata.get_sample_info(cxi_file)
        wavelength = cdtdata.get_wavelength(cxi_file)
        distance, basis, corner = cdtdata.get_detector_geometry(cxi_file)
        detector_geometry = {'distance' : distance,
                             'basis'    : basis,
                             'corner'   : corner}
        mask = cdtdata.get_mask(cxi_file)
        patterns, axes = cdtdata.get_data(cxi_file)

        translations = cdtdata.get_ptycho_translations(cxi_file)
        return cls(translations, patterns, axes=axes,
                   entry_info = entry_info,
                   sample_info = sample_info,
                   wavelength=wavelength,
                   detector_geometry=detector_geometry,
                   mask=mask)
    

    def to_cxi(self, cxi_file):
        super(Ptycho_2D_Dataset,self).to_cxi(cxi_file)
        cdtdata.add_data(cxi_file, self.patterns, axes=self.axes)
        cdtdata.add_ptycho_translations(cxi_file, self.translations)

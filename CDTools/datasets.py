from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from copy import copy

# The naming overlap here is definitely going to get confusing.
from CDTools.tools import data as cdtdata
from torch.utils import data as torchdata

__all__ = ['CDataset', 'Ptycho_2D_Dataset']

#
# I think that here should live a variety of datasets that all
# subclass the basic dataset class. They should be able to contain
# all the information that would be available in a CDI-type experiment.
# 
# It's important for each kind of dataset to have tools to:
#
# * Pass too and from the GPU (where to store the data)
# * Be able to load itself intelligently from a cxi file
# * Be able to save itself intelligently to a cxi file
#
# And in addition, it should be relatively simple to initialize from
# data held as python structures
#
# The datasets need to be able to work just like pytorch datasets,
# in fact, they should subclass them, where they return the relevant
# information for each diffraction pattern when sliced.
#


#
# One question, though, is how to deal with information that models need
# to know as well as the datasets? The basic issue is that it would be
# nice to be able to write down a forward model and have it populate
# a dataset object with the simulated data. But you also want to be able
# to write a forward model that pulls the relevant information from a
# dataset that has been read from a real data's cxi file.
#
# One workaround would be to just have the things defined twice, but to
# make sure that every model can load/initialize itself from a dataset.
# If it also knows how to define itself from an initialization function
# or a python dataset, then it can easily also have a generic function to
# simulate the action of the model and create a dataset from that simulation.
#
# This is unrelated, but it will then be important to be able to save and load
# models easily from a predefined format. To be honest, this could just
# literally be a snippet of python code that redefines the specific model
# using only the base CDTools packages. That way a saved model could be a
# .py file that just creates the model on the spot. Not sure I love this
# though
#
# Is the trio of data, model, and reconstruction the correct thing though?
# Perhaps the reconstruction doesn't need to exist. The model can have
# a few more things involved in it, and the reconstructions could be a
# single function, for example, that only uses what's in the model.
#
# The issue with the automatic differentiation stuff is that it's history
# dependent in a sense. So you generate an optimization object and you take
# a step with it.
#
# Can all reconstruction algorithms be formulated in terms of an explicit
# forward step, backward step, and gradient with respect to the forward
# step? No, for example the position annealing step needs to do it's own
# thing. Okay, well then it could be possible to write an automatic
# differentiation sequence that can generate n steps of a given automatic
# differentiation solved. So given a model, you could just generically
# create n steps of automatic differentiation. But then also write an
# explicit ePIE step, etc. It could be interesting if the reconstruction
# algorithms defined generators which would run one step and then yield
# the loss at the end of that step. This would let you easily save out
# the loss but also would make it simple to include live plotting
#
# 
# Then instead of a plan, you would just write a python script to follow
# the steps you wanted to follow. 
#
#


# I think there should be a base CDataSet that has entry_info, sample_info, detector_info, and mask attributes. It can know how to load this data from a cxi file and write it out to a cxi file. It also can know how to pass this info to and from the GPU (not all of it needs to be passed in that way).

# Then, I will write a few common datasets as a demonstration. First is
# A 2D CDI dataset, second is a 2D Ptycho dataset. 


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

            
    def to(self,*args,**kwargs):
        # The mask should always stay a uint8, but it should switch devices
        mask_kwargs = copy(kwargs)
        try:
            mask_kwargs.pop('dtype')
        except KeyError as r:
            pass
        
        if self.mask is not None:
            self.mask = self.mask.to(*args,**mask_kwargs)          

    
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

    def __getitem__(self, index):
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

from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from matplotlib import pyplot as plt
from CDTools.datasets import CDataset
from CDTools.tools import data as cdtdata


__all__ = ['BasicPtychoDataset']


class BasicPtychoDataset(CDataset):
    """The standard dataset for a 2D ptychography scan"""

    def __init__(self, translations, patterns, *args, **kwargs):
        """Initialize the dataset from python objects"""

        super(BasicPtychoDataset,self).__init__(*args, **kwargs)
        self.translations = t.Tensor(translations).clone()
        self.patterns = t.Tensor(patterns).clone()

    def __len__(self):
        return self.patterns.shape[0]

    def _load(self, index):
        return (index, self.translations[index]), self.patterns[index]


    def to(self, *args, **kwargs):
        """Sends the relevant data to the given device and dtype"""
        super(BasicPtychoDataset,self).to(*args,**kwargs)
        self.translations = self.translations.to(*args, **kwargs)
        self.patterns = self.patterns.to(*args, **kwargs)


    @classmethod
    def from_cxi(cls, cxi_file):
        """Generates a new CDataset from a .cxi file directly"""

        # Generate a base dataset
        dataset = CDataset.from_cxi(cxi_file)
        # Mutate the class to this subclass (BasicPtychoDataset)
        dataset.__class__ = cls

        # Load the data that is only relevant for this class
        patterns, axes = cdtdata.get_data(cxi_file)
        translations = cdtdata.get_ptycho_translations(cxi_file)

        # And now re-add it
        dataset.translations = t.Tensor(translations).clone()
        dataset.patterns = t.Tensor(patterns).clone()

        return dataset

    
    def to_cxi(self, cxi_file):
        """Saves out a BasicPtychoDataset as a .cxi file"""

        super(BasicPtychoDataset,self).to_cxi(cxi_file)
        cdtdata.add_data(cxi_file, self.patterns, axes=self.axes)
        cdtdata.add_ptycho_translations(cxi_file, self.translations)


    def inspect(self):
        """Plots a random diffraction pattern"""

        index = np.random.randint(len(self))
        plt.figure()
        plt.imshow(self.patterns[index,:,:].cpu().numpy())

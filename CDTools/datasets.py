from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from copy import copy

from CDTools.tools import data as cdtdata
from CDTools.tools import plotting
from torch.utils import data as torchdata
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import ticker

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
        super(Ptycho_2D_Dataset,self).to_cxi(cxi_file)
        cdtdata.add_data(cxi_file, self.patterns, axes=self.axes)
        cdtdata.add_ptycho_translations(cxi_file, self.translations)


    def inspect(self):
        fig, axes = plt.subplots(1,2,figsize=(8,5.3))
        fig.tight_layout(rect=[0.04, 0.09, 0.98, 0.96])
        axslider = plt.axes([0.15,0.06,0.75,0.03])

        translations = self.translations.detach().cpu().numpy()
        nanomap_values = self.patterns.sum(dim=(1,2)).detach().cpu().numpy()
        
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
                print('hi')
                axes[0].set_title('Nanomap')
                axes[1].set_title('Pattern')

                bbox = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                
                s = bbox.width * bbox.height / translations.shape[0] * 72**2 #72 is points per inch
                s /= 4 # A rough value to make the size work out
                s = np.ones(len(nanomap_values)) * s

                s[idx] *= 4
    
                nanomap = axes[0].scatter(1e6 * translations[:,0],1e6 * translations[:,1],s=s,c=nanomap_values)
    
                axes[0].invert_xaxis()
                axes[0].set_facecolor('k')
                axes[0].set_xlabel('Translation x (um)')
                axes[0].set_ylabel('Translation y (um)')
                cb1 = plt.colorbar(nanomap, ax=axes[0], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.15,fraction=0.1)
                cb1.ax.tick_params(labelrotation=20)
    
                meas = axes[1].imshow(meas_data * mask)

                cb2 = plt.colorbar(meas, ax=axes[1], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.15,fraction=0.1)
                cb2.ax.tick_params(labelrotation=20)
                cb2.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(meas))
                
            else:
                axes[0].set_title('Nanomap')
                bbox = axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                
                s = bbox.width * bbox.height / translations.shape[0] * 72**2 #72 is points per inch
                s /= 4 # A rough value to make the size work out
                s = np.ones(len(nanomap_values)) * s
                s[idx] *= 4
    
                axes[0].clear()
                nanomap = axes[0].scatter(1e6 * translations[:,0],1e6 * translations[:,1],s=s,c=nanomap_values)
                axes[0].invert_xaxis()
                axes[0].set_facecolor('k')
                axes[0].set_xlabel('Translation x (um)')
                axes[0].set_ylabel('Translation y (um)')


                
                meas = axes[1].images[-1]
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
        


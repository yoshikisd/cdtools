import torch as t
from cdtools.models import CDIModel
from cdtools.datasets import Ptycho2DDataset
from cdtools import tools
from cdtools.tools import plotting as p
from copy import copy
from torch.utils import data as torchdata
from datetime import datetime
import numpy as np

__all__ = ['SimplePtycho']        


class SimplePtycho(CDIModel):
    """A simple ptychography model for exploring ideas and extensions



    """
    def __init__(self, wavelength, detector_geometry,
                 probe_basis, detector_slice,
                 probe_guess, obj_guess, min_translation = [0,0],
                 surface_normal=np.array([0.,0.,1.]), mask=None):

        super(SimplePtycho,self).__init__()
        self.register_buffer('wavelength', t.as_tensor(wavelength))
        self.store_detector_geometry(detector_geometry)

        self.register_buffer('min_translation', t.as_tensor(min_translation))
        self.register_buffer('probe_basis', t.as_tensor(probe_basis))
        self.detector_slice = copy(detector_slice)

        self.register_buffer('surface_normal', t.as_tensor(surface_normal))

        
        if mask is None:
            self.register_buffer('mask', None)
        else:
            self.register_buffer('mask', t.as_tensor(mask, dtype=t.bool))

        probe_guess = t.tensor(probe_guess, dtype=t.complex64)
        obj_guess = t.tensor(obj_guess, dtype=t.complex64)

        # We rescale the probe here so it learns at the same rate as the
        # object
        self.register_buffer('probe_norm', t.max(t.abs(probe_guess)))

        self.probe = t.nn.Parameter(probe_guess / self.probe_norm)
        self.obj = t.nn.Parameter(obj_guess)


    @classmethod
    def from_dataset(cls, dataset):
        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']

        # always do this on the cpu
        get_as_args = dataset.get_as_args
        dataset.get_as(device='cpu')
        (indices, translations), patterns = dataset[:]
        dataset.get_as(*get_as_args[0],**get_as_args[1])

        center = tools.image_processing.centroid(t.sum(patterns,dim=0))

        # Then, generate the probe geometry from the dataset
        ewg = tools.initializers.exit_wave_geometry
        probe_basis, probe_shape, det_slice =  ewg(det_basis,
                                                   det_shape,
                                                   wavelength,
                                                   distance,
                                                   center=center)

        if hasattr(dataset, 'sample_info') and \
           dataset.sample_info is not None and \
           'orientation' in dataset.sample_info:
            surface_normal = dataset.sample_info['orientation'][2]
        else:
            surface_normal = np.array([0.,0.,1.])

        # Next generate the object geometry from the probe geometry and
        # the translations
        pix_translations = tools.interactions.translations_to_pixel(
            probe_basis,
            translations,
            surface_normal=surface_normal)
        obj_size, min_translation = tools.initializers.calc_object_setup(
            probe_shape,
            pix_translations)

        # Finally, initialize the probe and  object using this information
        probe = tools.initializers.SHARP_style_probe(dataset, probe_shape, det_slice)
        
        obj = t.ones(obj_size).to(dtype=t.complex64)
        det_geo = dataset.detector_geometry


        if hasattr(dataset, 'mask') and dataset.mask is not None:
            mask = dataset.mask.to(t.bool)
        else:
            mask = None

        return cls(wavelength,
                   det_geo,
                   probe_basis,
                   det_slice,
                   probe,
                   obj,
                   min_translation=min_translation,
                   mask=mask,
                   surface_normal=surface_normal)


    def interaction(self, index, translations):
        pix_trans = tools.interactions.translations_to_pixel(
            self.probe_basis,
            translations,
            surface_normal=self.surface_normal)
        
        pix_trans -= self.min_translation
        return tools.interactions.ptycho_2D_round(
            self.probe_norm * self.probe,
            self.obj,
            pix_trans)


    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)


    def backward_propagator(self, wavefields):
        return tools.propagators.inverse_far_field(wavefields)


    def measurement(self, wavefields):
        return tools.measurements.intensity(wavefields,
                                            detector_slice=self.detector_slice)
    

    def loss(self, real_data, sim_data, mask=None):
        return tools.losses.amplitude_mse(real_data, sim_data, mask=mask)


    plot_list = [
        ('Probe Amplitude',
         lambda self, fig: p.plot_amplitude(self.probe, fig=fig, basis=self.probe_basis)),
        ('Probe Phase',
         lambda self, fig: p.plot_phase(self.probe, fig=fig, basis=self.probe_basis)),
        ('Object Amplitude',
         lambda self, fig: p.plot_amplitude(self.obj, fig=fig, basis=self.probe_basis)),
        ('Object Phase',
         lambda self, fig: p.plot_phase(self.obj, fig=fig, basis=self.probe_basis))
    ]
    

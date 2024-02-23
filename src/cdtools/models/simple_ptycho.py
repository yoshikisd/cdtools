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
    def __init__(
            self,
            wavelength,
            detector_geometry,
            probe_basis,
            probe_guess,
            obj_guess,
            min_translation = [0,0],
    ):

        # We initialize the superclass
        super(SimplePtycho,self).__init__()

        # We register all the constants, like wavelength, as buffers.
        # This lets the model hook into some nice pytorch features, like
        # using model.to, and in principle should make easier to broadcast
        # across multiple GPUs
        self.register_buffer('wavelength', t.as_tensor(wavelength))
        self.store_detector_geometry(detector_geometry)

        self.register_buffer('min_translation', t.as_tensor(min_translation))
        self.register_buffer('probe_basis', t.as_tensor(probe_basis))

        probe_guess = t.tensor(probe_guess, dtype=t.complex64)
        obj_guess = t.tensor(obj_guess, dtype=t.complex64)

        # We rescale the probe here so it learns at the same rate as the
        # object
        self.register_buffer('probe_norm', t.max(t.abs(probe_guess)))

        # And we store the probe and object guesses as parameters, so
        # they can get optimized by pytorch
        self.probe = t.nn.Parameter(probe_guess / self.probe_norm)
        self.obj = t.nn.Parameter(obj_guess)


    @classmethod
    def from_dataset(cls, dataset):

        # We get the key geometry information from the dataset
        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']

        # We extract all the diffraction patterns and translations
        (indices, translations), patterns = dataset[:]

        # Then, we generate the probe geometry
        ewg = tools.initializers.exit_wave_geometry
        probe_basis =  ewg(det_basis, det_shape, wavelength, distance)

        # Next generate the object geometry from the probe geometry and
        # the translations
        pix_translations = tools.interactions.translations_to_pixel(
            probe_basis,
            translations,
            surface_normal=surface_normal)
        obj_size, min_translation = tools.initializers.calc_object_setup(
            probe_shape,
            pix_translations)

        # Finally, initialize the probe and object using this information
        probe = tools.initializers.SHARP_style_probe(
            dataset,
            probe_shape,
            det_slice
        )
        obj = t.ones(obj_size).to(dtype=t.complex64)

        return cls(
            wavelength,
            dataset.detector_geometry,
            probe_basis,
            det_slice,
            probe,
            obj,
            min_translation=min_translation
        )


    def interaction(self, index, translations):
        
        # We map from real-space to pixel-space units
        pix_trans = tools.interactions.translations_to_pixel(
            self.probe_basis,
            translation)
        pix_trans -= self.min_translation

        # This function gets the proper, pixel-accurate crops of the
        # object array, and uses a sinc-interpolated subpixel shift of
        # the probe to reach subpixel accuracy
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


    def loss(self, real_data, sim_data):
        return tools.losses.amplitude_mse(real_data, sim_data)


    # This lists all the plots to display on a call to model.inspect()
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
    

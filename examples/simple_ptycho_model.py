from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from CDTools.models import CDIModel
from CDTools import tools

__all__ = ['SimplePtycho']

class SimplePtycho(CDIModel):
    """A simple ptychography model for exploring ideas and extensions"""
    
    def __init__(self, probe_basis, probe_guess, obj_guess,
                 min_translation = [0,0]):
        # We have to initialize the Module
        super(SimplePtycho,self).__init__()

        # We first save the relevant information
        self.min_translation = t.Tensor(min_translation)
        self.probe_basis = t.Tensor(probe_basis)

        # We rescale the probe so it learns at the same rate as the object
        self.probe_norm = t.max(tools.cmath.cabs(probe_guess.to(t.float32)))
        self.probe = t.nn.Parameter(probe_guess.to(t.float32)
                                    / self.probe_norm)
        self.obj = t.nn.Parameter(obj_guess.to(t.float32))



    @classmethod
    def from_dataset(cls, dataset):
        # First, load the information from the dataset
        wavelength = dataset.wavelength
        det_basis = dataset.detector_geometry['basis']
        det_shape = dataset[0][1].shape
        distance = dataset.detector_geometry['distance']
        (indices, translations), patterns = dataset[:]
        
        # Then, generate the probe geometry
        ewg = tools.initializers.exit_wave_geometry
        probe_basis, probe_shape, det_slice =  ewg(det_basis,
                                                   det_shape,
                                                   wavelength,
                                                   distance,
                                                   opt_for_fft=False)

        # Next generate the object geometry
        pix_translations = tools.interactions.translations_to_pixel(
            probe_basis, translations)
        obj_size, min_translation = tools.initializers.calc_object_setup(
            probe_shape, pix_translations)

        # Finally, initialize the probe and  object using this information
        probe = tools.initializers.SHARP_style_probe(dataset,
                                                     probe_shape,
                                                     det_slice)

        obj = t.ones(obj_size+(2,))

        return cls(probe_basis, probe, obj, min_translation=min_translation)
    

    def interaction(self, index, translations):
        pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                             translations)
        pix_trans -= self.min_translation
        return tools.interactions.ptycho_2D_round(self.probe_norm * self.probe,
                                                  self.obj,
                                                  pix_trans)

    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)

    def measurement(self, wavefields):
        return tools.measurements.intensity(wavefields)

    def loss(self, sim_data, real_data):
        return tools.losses.amplitude_mse(real_data, sim_data)


    def to(self, *args, **kwargs):
        super(SimplePtycho, self).to(*args, **kwargs)
        self.min_translation = self.min_translation.to(*args,**kwargs)
        self.probe_basis = self.probe_basis.to(*args,**kwargs)
        self.probe_norm = self.probe_norm.to(*args,**kwargs)

        
   
    plot_list = [
        ('Probe Amplitude',
         lambda self, fig: tools.plotting.plot_amplitude(self.probe, fig=fig, basis=self.probe_basis)),
        ('Probe Phase',
         lambda self, fig: tools.plotting.plot_phase(self.probe, fig=fig, basis=self.probe_basis)),
        ('Object Amplitude',
         lambda self, fig: tools.plotting.plot_amplitude(self.obj, fig=fig, basis=self.probe_basis)),
        ('Object Phase',
         lambda self, fig: tools.plotting.plot_phase(self.obj, fig=fig, basis=self.probe_basis))
    ]
    
    def save_results(self):
        probe = tools.cmath.torch_to_complex(self.probe.detach().cpu())
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = tools.cmath.torch_to_complex(self.obj.detach().cpu())
        return {'probe':probe,'obj':obj}

    
if __name__ == '__main__':
    

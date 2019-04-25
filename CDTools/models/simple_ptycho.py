from __future__ import division, print_function, absolute_import

import torch as t
from CDTools.models import CDIModel
from CDTools import tools
from CDTools.tools import plotting as p
from copy import copy
from torch.utils import data as torchdata
from matplotlib import pyplot as plt


class SimplePtycho(CDIModel):

    def __init__(self, wavelength, detector_geometry,
                 probe_basis, detector_slice,
                 probe_guess, obj_guess, min_translation = [0,0],
                 mask=None):

        super(SimplePtycho,self).__init__()
        self.wavelength = t.Tensor([wavelength])
        self.detector_geometry = copy(detector_geometry)
        det_geo = self.detector_geometry
        if hasattr(det_geo, 'distance'):
            det_geo['distance'] = t.Tensor(det_geo['distance'])
        if hasattr(det_geo, 'basis'):
            det_geo['basis'] = t.Tensor(det_geo['basis'])
        if hasattr(det_geo, 'corner'):
            det_geo['corner'] = t.Tensor(det_geo['corner'])

        self.min_translation = t.Tensor(min_translation)

        self.probe_basis = t.Tensor(probe_basis)
        self.detector_slice = detector_slice
        if mask is None:
            self.mask = None
        else:
            self.mask = t.ByteTensor(mask)

        # We rescale the probe here so it learns at the same rate as the
        # object
        self.probe_norm = t.max(tools.cmath.cabs(probe_guess.to(t.float32)))

        self.probe = t.nn.Parameter(probe_guess.to(t.float32)
                                    / self.probe_norm)
        self.obj = t.nn.Parameter(obj_guess.to(t.float32))


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

        # Next generate the object geometry from the probe geometry and
        # the translations
        pix_translations = tools.interactions.translations_to_pixel(probe_basis, translations)
        obj_size, min_translation = tools.initializers.calc_object_setup(probe_shape, pix_translations)

        # Finally, initialize the probe and  object using this information
        probe = tools.initializers.SHARP_style_probe(dataset, probe_shape, det_slice)

        obj = t.ones(obj_size+(2,))
        det_geo = dataset.detector_geometry

        if hasattr(dataset, 'mask') and dataset.mask is not None:
            mask = dataset.mask.to(t.uint8)
        else:
            mask = None

        return cls(wavelength, det_geo, probe_basis, det_slice, probe, obj, min_translation=min_translation, mask=mask)


    def interaction(self, index, translations):
        pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                             translations)
        pix_trans -= self.min_translation
        return tools.interactions.ptycho_2D_round(self.probe_norm * self.probe,
                                                  self.obj,
                                                  pix_trans)


    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)


    def backward_propagator(self, wavefields):
        return tools.propagators.inverse_far_field(wavefields)


    def measurement(self, wavefields):
        return tools.measurements.intensity(wavefields,
                                            detector_slice=self.detector_slice)


    def loss(self, sim_data, real_data, mask=None):
        return tools.losses.amplitude_mse(real_data, sim_data, mask=mask)


    def to(self, *args, **kwargs):
        super(SimplePtycho, self).to(*args, **kwargs)
        self.wavelength = self.wavelength.to(*args,**kwargs)
        # move the detector geometry too
        det_geo = self.detector_geometry
        if hasattr(det_geo, 'distance'):
            det_geo['distance'] = det_geo['distance'].to(*args,**kwargs)
        if hasattr(det_geo, 'basis'):
            det_geo['basis'] = det_geo['basis'].to(*args,**kwargs)
        if hasattr(det_geo, 'corner'):
            det_geo['corner'] = det_geo['corner'].to(*args,**kwargs)

        if self.mask is not None:
            self.mask = self.mask.to(*args, **kwargs)

        self.min_translation = self.min_translation.to(*args,**kwargs)
        self.probe_basis = self.probe_basis.to(*args,**kwargs)
        self.probe_norm = self.probe_norm.to(*args,**kwargs)


    def sim_to_dataset(self, args_list):
        raise NotImplementedError()


    def inspect(self):
        p.plot_amplitude(self.probe, basis=self.probe_basis)
        plt.title('Probe Amplitude')
        p.plot_phase(self.probe, basis=self.probe_basis)
        plt.title('Probe Phase')
        p.plot_amplitude(self.obj, basis=self.probe_basis)
        plt.title('Object Amplitude')
        p.plot_phase(self.obj, basis=self.probe_basis)
        plt.title('Object Phase')


    def save_results(self):
        probe = tools.cmath.torch_to_complex(self.probe.detach().cpu())
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = tools.cmath.torch_to_complex(self.obj.detach().cpu())
        return {'probe':probe,'obj':obj}
        
    
    def ePIE(self, iterations, dataset, beta = 1.0):
        """Runs an ePIE reconstruction as described in `Maiden et al. (2017) <https://www.osapublishing.org/optica/abstract.cfm?uri=optica-4-7-736>`_.
        Optional parameters are:

        :arg ``iterations``: Controls the number of iterations run, defaults to 1.
        :arg ``beta``: Algorithmic parameter described in Maiden's implementation of rPIE. Defaults to 0.15.
        :arg ``probe``: Initial probe wavefunction.
        :arg ``object``: Initial object wavefunction.
        """
        probe_shape = self.probe.shape

        if self.mask is not None:
            mask = self.mask[...,None]
        else:
            mask=None

        def probe_update(exit_wave, exit_wave_corrected, probe, object, translation):
            return probe+tools.cmath.cmult(beta*(tools.cmath.cconj(object)/t.max(tools.cmath.cabssq(object)))[translation[0]:translation[0]+probe_shape[0],translation[1]:translation[1]+probe_shape[1]], \
            exit_wave_corrected-exit_wave)

        def object_update(exit_wave, exit_wave_corrected, probe, object, translation):
            object[translation[0]:translation[0]+probe_shape[0],translation[1]:translation[1]+probe_shape[1]]\
            +=tools.cmath.cmult(tools.cmath.cconj(probe)/t.max(tools.cmath.cabssq(probe)),exit_wave_corrected-exit_wave)
            return object

        with t.no_grad():
            data_loader = torchdata.DataLoader(dataset, shuffle=True)

            for it in range(iterations):
                loss = []
                for (i, [translations]), [patterns] in data_loader:
                    probe = self.probe.clone()
                    object = self.obj.clone()
                    exit_wave = self.interaction(i, translations)
                    exit_wave_corrected = exit_wave.clone()
                    exit_wave_corrected[self.detector_slice] = tools.projectors.modulus(self.forward_propagator(exit_wave)[self.detector_slice], patterns, mask = mask)

                    integer_translations = t.round(translations).to(dtype=t.int32)
                    self.probe.data = probe_update(exit_wave, exit_wave_corrected, probe, object, integer_translations)
                    self.obj.data = object_update(exit_wave, exit_wave_corrected, probe, object, integer_translations)
                    loss.append(self.loss(self.measurement(self.interaction(i, translations)), patterns))

                yield t.mean(t.Tensor(loss)).cpu().numpy()

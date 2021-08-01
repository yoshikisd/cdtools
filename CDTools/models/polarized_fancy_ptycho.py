import torch as t
from CDTools.models import CDIModel, FancyPtycho
from CDTools.datasets import Ptycho2DDataset
from CDTools import tools
from CDTools.tools import plotting as p
from CDTools.tools import analysis
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
from scipy import linalg as sla
from copy import copy
from CDTools.tools import polarization

__all__ = ['PolarizedFancyPtycho']

class PolarizedFancyPtycho(FancyPtycho):

    def __init__(self, wavelength, detector_geometry,
                 probe_basis,
                 probe_guess, obj_guess, polarizer, analyzer,
                 detector_slice=None,
                 surface_normal=np.array([0.,0.,1.]),
                 min_translation = t.Tensor([0,0]),
                 background = None, translation_offsets=None, 
                 polarizer_offsets=None, analyzer_offsets=None,
                 polarizer_scale=1, analyzer_scale=1,  mask=None,
                 weights = None, translation_scale = 1, saturation=None,
                 probe_support = None, obj_support=None, oversampling=1,
                 loss='amplitude mse',units='um'):
        
        super(FancyPtycho, self).__init__(wavelength, detector_geometry,
                 probe_basis,
                 probe_guess, obj_guess,
                 detector_slice=None,
                 surface_normal=np.array([0.,0.,1.]),
                 min_translation = t.Tensor([0,0]),
                 background = None, translation_offsets=None, mask=None,
                 weights = None, translation_scale = 1, saturation=None,
                 probe_support = None, obj_support=None, oversampling=1,
                 loss='amplitude mse',units='um')

        if polarizer_offsets is None:
            self.polarizer_offsets = None
        else:
            self.polarizer_offsets = t.nn.Parameter(t.tensor(polarizer_offsets).to(dtype=t.float32)) / polarizer_scale

        if analyzer_offsets is None:
            self.analyzer_offsets = None
        else:
            self.analyzer_offsets = t.nn.Parameter(t.tensor(analyzer_offsets).to(dtype=t.float32)) / analyzer_scale
        
    @classmethod
    def from_dataset(cls, dataset, probe_size=None, randomize_ang=0, padding=0, n_modes=1, dm_rank=None, translation_scale = 1, saturation=None, probe_support_radius=None, propagation_distance=None, restrict_obj=-1, scattering_mode=None, oversampling=1, auto_center=False, opt_for_fft=False, loss='amplitude mse', units='um', left_polarized=True):
        
        model = FancyPtycho.from_dataset(dataset, probe_size=None, randomize_ang=0, padding=0, n_modes=1, dm_rank=None, translation_scale = 1, saturation=None, probe_support_radius=None, propagation_distance=None, restrict_obj=-1, scattering_mode=None, oversampling=1, auto_center=False, opt_for_fft=False, loss='amplitude mse', units='um', polarized=True)


        # Mutate the class to its subclass 
        model.__class__ = cls

        if left_polarized:
            x = 1j
        else:
            x = -1j
        model.probe.data = t.stack((model.probe.data.to(dtype=t.cfloat), x * model.probe.data.to(dtype=t.cfloat)), dim=-3)
        obj = t.stack((model.obj.data, model.obj.data), dim=-3)
        model.obj.data = t.stack((obj, obj), dim=-4)        

        # tensor vs tensor.data
        return model

    
    # WHAT IS INDEX?
    def interaction(self, index, translations, polarizer, analyzer, test=False):

        # Step 1 is to convert the translations for each position into a
        # value in pixels
        pix_trans = tools.interactions.translations_to_pixel(self.probe_basis,
                                                             translations,
                                                             surface_normal=self.surface_normal)
        pix_trans -= self.min_translation
        # We then add on any recovered translation offset, if they exist
        if self.translation_offsets is not None:
            pix_trans += self.translation_scale * self.translation_offsets[index]

             
        # This restricts the basis probes to stay within the probe support
        basis_prs = self.probe * self.probe_support[...,:,:] # This makes no sense 
        # self.probe is an Nx2xXxY stach of probes

        # Now we construct the probes for each shot from the basis probes
        Ws = self.weights[index]
        if len(self.weights[0].shape) == 0:
            # If a purely stable coherent illumination is defined
            # Ws is a tensor of length M, M is the number of frames to be processed
            prs = Ws[...,None,None,None,None] * basis_prs
        else:
            raise NotImplementedError('Unstable Modes not Implemented for polarized light')
            
        pol_probes = polarization.apply_linear_polarizer(prs, polarizer)

        exit_waves = self.probe_norm * tools.interactions.ptycho_2D_sinc(
            prs, self.obj_support * self.obj,pix_trans,
            shift_probe=True, multiple_modes=True, polarized=True)

        analyzed_exit_waves = polarization.apply_linear_polarizer(exit_waves, analyzer)

        return analyzed_exit_waves
    

    def vectorial_wavefields(wavefields, func, *args, **kwargs):
        wavefields_x = wavefields[..., 0, :, :, :]
        wavefields_y = wavefields[..., 1, :, :, :]
        out_x = func(wavefields_x, *args, **kwargs)
        out_y = func(wavefields_y, *args, **kwargs)
        out = t.stack((out_x, out_y), dim=-4)
        return out[..., None, :, :]

    def forward_propagator(self, wavefields):
        return tools.propagators.far_field(wavefields)

    def backward_propagator(self, wavefields):
        return tools.propagators.inverse_far_field(wavefields)

    
    def measurement(self, wavefields):
        wavefields_x = wavefields[..., 0, :, :]
        wavefields_y = wavefields[..., 1, :, :]
        out_x = tools.measurements.quadratic_background(wavefields_x,
                            self.background,
                            detector_slice=self.detector_slice,
                            measurement=tools.measurements.incoherent_sum,
                            saturation=self.saturation,
                            oversampling=self.oversampling)
        # now, set bckgr to 0 since t shouldn't be calculated twice
        out_y = tools.measurements.quadratic_background(wavefields_y,
                            0,
                            detector_slice=self.detector_slice,
                            measurement=tools.measurements.incoherent_sum,
                            saturation=self.saturation,
                            oversampling=self.oversampling)

        return out_x + out_y


    # Note: No "loss" function is defined here, because it is added
    # dynamically during object creation in __init__
    
    def to(self, *args, **kwargs):
        super(PolarizedFancyPtycho, self).to(*args, **kwargs)

        
    def sim_to_dataset(self, args_list):
        # In the future, potentially add more control
        # over what metadata is saved (names, etc.)
        
        # First, I need to gather all the relevant data
        # that needs to be added to the dataset
        entry_info = {'program_name': 'CDTools',
                      'instrument_n': 'Simulated Data',
                      'start_time': datetime.now()}

        surface_normal = self.surface_normal.detach().cpu().numpy()
        xsurfacevec = np.cross(np.array([0.,1.,0.]), surface_normal)
        xsurfacevec /= np.linalg.norm(xsurfacevec)
        ysurfacevec = np.cross(surface_normal, xsurfacevec)
        ysurfacevec /= np.linalg.norm(ysurfacevec)
        orientation = np.array([xsurfacevec, ysurfacevec, surface_normal])
        
        sample_info = {'description': 'A simulated sample',
                       'orientation': orientation}

        
        detector_geometry = self.detector_geometry
        mask = self.mask
        wavelength = self.wavelength
        indices, translations = args_list
        
        # Then we simulate the results
        data = self.forward(indices, translations)

        # And finally, we make the dataset
        return Ptycho2DDataset(translations, data,
                                 entry_info = entry_info,
                                 sample_info = sample_info,
                                 wavelength=wavelength,
                                 detector_geometry=detector_geometry,
                                 mask=mask)

    
    def corrected_translations(self, dataset):
        translations = dataset.translations.to(dtype=t.float32,device=self.probe.device)
        t_offset = tools.interactions.pixel_to_translations(self.probe_basis,self.translation_offsets*self.translation_scale,surface_normal=self.surface_normal)
        return translations + t_offset


    
    def get_rhos(self):
        # If this is the general unified mode model
        if self.weights.dim() >= 2:
            Ws = self.weights.detach().cpu().numpy()
            rhos_out = np.matmul(np.swapaxes(Ws,1,2), Ws.conj())
            return rhos_out
        # This is the purely incoherent case
        else:
            return np.array([np.eye(self.probe.shape[0])]*self.weights.shape[0],
                            dtype=np.complex64)
        
    def tidy_probes(self, normalization=1, normalize=False):
        """Tidies up the probes
        
        What we want to do here is use all the information on all the probes
        to calculate a natural basis for the experiment, and update all the
        density matrices to operate in that updated basis
        
        """
        
        # First we treat the purely incoherent case
        
        # I don't love this pattern of using an if statement with a return
        # to catch this case, but because it's so much simpler than the
        # unified mode case I think it's appropriate
        if self.weights.dim() == 1:
            probe = self.probe.detach().cpu().numpy()
            ortho_probes = analysis.orthogonalize_probes(probe)
            self.probe.data = t.as_tensor(ortho_probes,
                    device=self.probe.device,dtype=self.probe.dtype)
            return

        # This is for the unified mode case
        
        # Note to future: We could probably do this more cleanly with an
        # SVD directly on the Ws matrix, instead of an eigendecomposition
        # of the rho matrix.
        
        rhos = self.get_rhos()
        overall_rho = np.mean(rhos,axis=0)
        probe = self.probe.detach().cpu().numpy()
        ortho_probes, A = analysis.orthogonalize_probes(probe,
                                        density_matrix=overall_rho,
                                        keep_transform=True,
                                        normalize=normalize)
        Aconj = A.conj()
        Atrans = np.transpose(A)
        new_rhos = np.matmul(Atrans,np.matmul(rhos,Aconj))

        new_rhos /= normalization
        ortho_probes *= np.sqrt(normalization)

        dm_rank = self.weights.shape[1]
        
        new_Ws = []
        for rho in new_rhos:
            # These are returned from smallest to largest - we want to keep
            # the largest ones
            w,v = sla.eigh(rho)
            w = w[::-1][:dm_rank]
            v = v[:,::-1][:,:dm_rank]
            # For situations where the rank of the density matrix is not
            # full in reality, but we keep more modes around than needed,
            # some ws can go negative due to numerical error! This is
            # extremely rare, but comon enough to cause crashes occasionally
            # when there are thousands of individual matrices to transform
            # every time this is called.
            w = np.maximum(w,0)
            
            new_Ws.append(np.dot(np.diag(np.sqrt(w)),v.transpose()))
            
        new_Ws = np.array(new_Ws)

        self.weights.data = t.as_tensor(new_Ws,
            dtype=self.weights.dtype,device=self.weights.device)
        
        self.probe.data = t.as_tensor(ortho_probes,
            device=self.probe.device,dtype=self.probe.dtype)


    def plot_wavefront_variation(self, dataset,fig=None,mode='amplitude',**kwargs):
        def get_probes(idx):
            basis_prs = self.probe * self.probe_support[...,:,:]
            prs = t.sum(self.weights[idx,:,:,None,None] * basis_prs, axis=-4)
            ortho_probes = analysis.orthogonalize_probes(prs)

            if mode.lower() == 'amplitude':
                return np.abs(ortho_probes.detach().cpu().numpy())
            if mode.lower() == 'root_sum_intensity':
                return np.sum(np.abs(ortho_probes.detach().cpu().numpy())**2,axis=0)
            if mode.lower() == 'phase':
                return np.angle(ortho_probes.detach().cpu().numpy())
            
        probe_matrix = np.zeros([self.probe.shape[0]]*2,
                                dtype=np.complex64)
        np_probes = self.probe.detach().cpu().numpy()
        for i in range(probe_matrix.shape[0]):
            for j in range(probe_matrix.shape[0]):
                probe_matrix[i,j] = np.sum(np_probes[i]*np_probes[j].conj())
        

        weights = self.weights.detach().cpu().numpy()
        
        probe_intensities = np.sum(np.tensordot(weights,probe_matrix,axes=1)*
                                   weights.conj(),axis=2)

        # Imaginary part is already essentially zero up to rounding error
        probe_intensities = np.real(probe_intensities)
        
        values = np.sum(probe_intensities,axis=1)
        if mode.lower() == 'amplitude' or mode.lower() == 'root_sum_intensity':
            cmap = 'viridis'
        else:
            cmap = 'twilight'
            
        p.plot_nanomap_with_images(self.corrected_translations(dataset), get_probes, values=values, fig=fig, units=self.units, basis=self.probe_basis, nanomap_colorbar_title='Total Probe Intensity',cmap=cmap,**kwargs),

        
    plot_list = [
        ('',
         lambda self, fig, dataset: self.plot_wavefront_variation(dataset,fig=fig,mode='root_sum_intensity',image_title='Root Summed Probe Intensities',image_colorbar_title='Square Root of Intensity'),
         lambda self: len(self.weights.shape) >= 2),
        ('',
         lambda self, fig, dataset: self.plot_wavefront_variation(dataset,fig=fig,mode='amplitude',image_title='Probe Amplitudes (scroll to view modes)',image_colorbar_title='Probe Amplitude'),
         lambda self: len(self.weights.shape) >= 2),
        ('',
         lambda self, fig, dataset: self.plot_wavefront_variation(dataset,fig=fig,mode='phase',image_title='Probe Phases (scroll to view modes)',image_colorbar_title='Probe Phase'),
         lambda self: len(self.weights.shape) >= 2),
        ('Basis Probe Amplitudes (scroll to view modes)',
         lambda self, fig: p.plot_amplitude(self.probe, fig=fig, basis=self.probe_basis,units=self.units)),
        ('Basis Probe Phases (scroll to view modes)',
         lambda self, fig: p.plot_phase(self.probe, fig=fig, basis=self.probe_basis,units=self.units)),
        ('Average Density Matrix Amplitudes',
         lambda self, fig: p.plot_amplitude(np.nanmean(np.abs(self.get_rhos()),axis=0), fig=fig),
         lambda self: len(self.weights.shape) >=2),
        ('% Power in Top Mode (only accurate after tidy_probes)',
         lambda self, fig, dataset: p.plot_nanomap(self.corrected_translations(dataset), analysis.calc_top_mode_fraction(self.get_rhos()), fig=fig,units=self.units),
         lambda self: len(self.weights.shape) >=2),
        ('Object Amplitude', 
         lambda self, fig: p.plot_amplitude(self.obj, fig=fig, basis=self.probe_basis,units=self.units)),
        ('Object Phase',
         lambda self, fig: p.plot_phase(self.obj, fig=fig, basis=self.probe_basis,units=self.units)),
        ('Corrected Translations',
         lambda self, fig, dataset: p.plot_translations(self.corrected_translations(dataset), fig=fig,units=self.units)),
        ('Background',
         lambda self, fig: plt.figure(fig.number) and plt.imshow(self.background.detach().cpu().numpy()**2))
    ]

    
    def save_results(self, dataset):
        basis = self.probe_basis.detach().cpu().numpy()
        translations = self.corrected_translations(dataset).detach().cpu().numpy()
        probe = self.probe.detach().cpu().numpy()
        probe = probe * self.probe_norm.detach().cpu().numpy()
        obj = self.obj.detach().cpu().numpy()
        background = self.background.detach().cpu().numpy()**2
        weights = self.weights.detach().cpu().numpy()
        
        return {'basis':basis, 'translation':translations,
                'probe':probe,'obj':obj,
                'background':background,
                'weights':weights}

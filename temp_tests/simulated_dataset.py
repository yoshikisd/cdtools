import numpy as np
import torch as t
from CDTools.models import PolarizedFancyPtycho
#from CDTools.datasets import Polarized2DDataset
import CDTools
from CDTools.tools import polarization
from CDTools import tools
from matplotlib import pyplot as plt
from PIL import Image

# upolad 4 different images representing 4 components of the object
# and 2 gaaussian functionas corresponding to the probe components

a = np.asarray(Image.open('a.jpg'))
b = np.asarray(Image.open('b.jpg'))
c = np.asarray(Image.open('c.jpg'))
d = np.asarray(Image.open('d.jpg'))

#a = np.dot(a[..., :3], [.3, 6., .1])

def simulate_dataset(probe_size, obj_size, num_patt):
    translations = []
    xs, ys = np.mgrid[:num_patt, :num_patt]
    for x, y in zip(xs, ys):
        translations.append((x*10e-3, y*10e-3))

    translations = t.as_tensor(translations, dtype=t.float32)
    a = t.as_tensor(a, dtype=t.cfloat)
    a = t.tensordot(a, t.tensor([.3, .6, .1], dtype=t.cfloat), dims=([-1],[0]))[:obj_size, :obj_size]
    
    probe = tools.initializers.gaussian(np.array([probe_size, probe_size]), 50)
    wavefields = tools.interactions.ptycho_2D_sinc(probe, obj, translations)
    patterns = tools.propagators.far_field(wavefront)
    patterns = np(patterns)
    translations = np(t.cat((translations, t.zeros(num_patt)), dim=-1))

    # needs to be stored as a cxi file
    dataset = CDTools.datasets.Ptycho2DDataset.from_cxi('simulated_dataset.cxi')
    dataset.detector_geometry = None

def simulate polarized_datset(probe_size, obj_size, num_patt):
    a, b, c, d = t.as_tensor(a, dtype=t.cfloat), t.as_tensor(b, dtype=t.cfloat), t.as_tensor(c, dtype=t.cfloat), t.as_tensor(d, dtype=t.cfloat)
    a = t.tensordot(a, t.tensor([.3, .6, .1], dtype=t.cfloat), dims=([-1],[0]))[:obj_size, :obj_size]
    b = t.tensordot(b, t.tensor([.3, .6, .1], dtype=t.cfloat), dims=([-1], [0]))[:obj_size, :obj_size]
    c = t.tensordot(c, t.tensor([.3, .6, .1], dtype=t.cfloat), dims=([-1], [0]))[:obj_size, :obj_size]
    d = t.tensordot(d, t.tensor([.3, .6, .1], dtype=t.cfloat), dims=([-1], [0]))[:obj_size, :obj_size]
    
    translations = []
    xs, ys = np.mgrid[:num_patt, :num_patt]
    for x, y in zip(xs, ys):
        translations.append((x*10e-3, y*10e-3))
    translations = t.as_tensor(translations, dtype=t.float32)
    
    obj = t.stack((t.stack((a, c), dim=0), t.stack((b, d), dim=0)), dim=-3)
    probe = tools.initializers.gaussian(np.array([probe_size, probe_size]), 50)
    probe = t.stack((probe, probe), dim=-3)
    probe = polarization.apply_circular_polarizer(probe)

    selections = tools.interactions.ptycho_2D_sinc(t.ones(2, probe_size, probe_size).to(dtype=t.cfloat), obj, translations, polarized=True)
    polarizers = [polarization.generate_linear_polarizer(i * 45) for i in range(3)]
    
    pol_probes = [polarization.apply_jones_matrix(probe, polarizers[i]) for i in range(3)]
    

    analyzer = t.stack(([polaryzers[i % 3] for i in range(num_patt)]), dim=0)
    # probes = t.stack(([probes[i // 3] for i in num_patt]), dim=0)
    wavefields = t.as_tensor([tools.interactions.ptycho_2D_sinc(pol_probes[i], obj, translations, polarized=True) for i in range(3)]).to(dtype=t.cfloat)

    wf = t.empty(1, 2, obj_size, obj_size)
    for i in range(num_patt):
        for j in range(3):
            pol_channel = t.stack(([wavefileds[j] for k in range(3)]), dim=0)
        wf = t.cat((wf, pol_channel), dim=0)
    
    pol_wavefieds = polarization.apply_jones_matrix(wf, analyzer)

    patterns = tools.propagators.far_field(pol_wavefieds)

    translations = np(t.cat((translations, t.zeros(num_patt)), dim=-1))
    patterns = np(patterns)
    dataset.detector_geometry = None
    # needs to be stored in a cxi file
    dataset = CDTools.datasets.FancyPtycho2DDataset.from_cxi('polarized_simulated_dataset.cxi')
    dataset.inspect()

    
    model = tools.models.PolarizedFancyPtycho.from_dataset(dataset)

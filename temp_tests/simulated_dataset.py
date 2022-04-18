import numpy as np
import torch as t
from scipy import misc
from CDTools.models import PolarizedFancyPtycho
from CDTools.datasets import PolarizedPtycho2DDataset
import CDTools
from CDTools.tools import polarization
from CDTools import tools
from matplotlib import pyplot as plt
from PIL import Image

# upolad 4 different images representing 4 components of the object
# and 2 gaaussian functionas corresponding to the probe components

f = misc.ascent()
x , y = np.shape(f)
aa = f[:x//2, :y//2]
bb = f[:x//2, -y//2:]
cc = f[-x//2:, :y//2]
dd = f[-x//2:, -y//2:]

print(1)

def simulate_polarized_dataset(probe_size, obj_size, num_patt, a, b, c, d):
    a, b, c, d = t.as_tensor(a, dtype=t.cfloat), t.as_tensor(b, dtype=t.cfloat), t.as_tensor(c, dtype=t.cfloat), t.as_tensor(d, dtype=t.cfloat)
    # a = t.tensordot(a, t.tensor([.3, .6, .1], dtype=t.cfloat), dims=([-1],[0]))[:obj_size, :obj_size]
    # b = t.tensordot(b, t.tensor([.3, .6, .1], dtype=t.cfloat), dims=([-1], [0]))[:obj_size, :obj_size]
    # c = t.tensordot(c, t.tensor([.3, .6, .1], dtype=t.cfloat), dims=([-1], [0]))[:obj_size, :obj_size]
    # d = t.tensordot(d, t.tensor([.3, .6, .1], dtype=t.cfloat), dims=([-1], [0]))[:obj_size, :obj_size]

    translations = []
    xs, ys = np.mgrid[:num_patt, :num_patt]
    xs, ys = np.ravel(xs), np.ravel(ys)
    for x, y in zip(xs, ys):
        for j in range(9):
            translations.append((5*x, 5*y))
    translations = t.as_tensor(translations, dtype=t.float32)
    obj = t.stack((t.stack((a, b), dim=-3), t.stack((c, d), dim=-3)), dim=-4)

    # for i, j in zip([0, 0, 1, 1], [0, 1, 0, 1]):
    #     plt.imshow(np.real(obj[i, j, ...]))
    #     plt.show()
    probe = tools.initializers.gaussian(np.array([probe_size, probe_size]), np.array((2, 2)))
    probe = t.stack((probe, probe), dim=-3)
    probe = polarization.apply_circular_polarizer(probe, multiple_modes=False)
    polarizers = polarization.generate_linear_polarizer(t.tensor([0, 45, 90]))
    num_transl = len(translations)
    pol_probes = [polarization.apply_jones_matrix(probe, polarizers[i], multiple_modes=False) for i in range(3)]
    analyzers = t.stack(([polarizers[i % 3] for i in range(num_transl)]), dim=0)
    analyzer = t.tensor([(i % 3) * 45 for i in range(num_transl)])
    polarizer = t.tensor([(i // 3) % 3 * 45 for i in range(num_transl)])
    polarized_probes = t.stack(([pol_probes[(i // 3) % 3] for i in range(num_transl)]), dim=0)
    polarized_wavefields = t.stack([tools.interactions.ptycho_2D_sinc(polarized_probes[i], obj, translations[i], polarized=True, multiple_modes=False) for i in range(num_transl)])
    wavefields = tools.propagators.far_field(polarized_wavefields)
    wavefields = polarization.apply_jones_matrix(wavefields, analyzers, multiple_modes=False)
    patterns = t.abs(wavefields[:, 0, :, :])**2 + t.abs(wavefields[:, 1, :, :])**2
    detector_basis = t.transpose(t.tensor([[0, -4.8e-6, 0], [-4.8e-6, 0, 0]]), 0, 1)
    det_shape = t.Size((obj_size, obj_size))
    wavelength = 532e-9
    real_basis = tools.initializers.exit_wave_geometry(detector_basis, det_shape, wavelength, 2.5e-2)[0]
    # print('f', type(real_basis), translations.shape)
    real_translations = tools.interactions.pixel_to_translations(real_basis, translations)
    # print(type(real_translations))
    patterns = patterns.numpy()
    detector_geometry = {
                         'corner': np.array([probe_size*4.8e-6/2, probe_size*4.8e-6/2, 2.5e-2]),
                         'basis': np.array([[0, -4.8e-6, 0], [-4.8e-6, 0, 0]]).transpose(),
                         'distance': 2.5e-2
                        }

    return PolarizedPtycho2DDataset(real_translations, polarizer, analyzer, patterns,
                                    axes=("x", "y"), detector_geometry=detector_geometry, wavelength=wavelength)

# print(1)
dataset = simulate_polarized_dataset(100, 500, 10, aa, bb, cc, dd)
# print(2)
dataset.inspect()
# print(3)
plt.show()
model = PolarizedFancyPtycho.from_dataset(dataset, propagation_distance=1e-3)
model.inspect()
# plt.show()
model.compare(dataset)
plt.show()

# for loss in model.Adam_optimize(400, dataset, batch_size=5, lr=0.002, schedule=True):
#     # And we liveplot the updates to the model as they happen
#     print(model.report())
#     model.inspect(dataset)
#     model.save_figures(prefix='simulated', extension='png')
#     res = model.save_results(dataset)
#     np.save('simulated_dataset.npy', res)

dataset = simulate_polarized_dataset(50, 100, 10, aa, bb, cc, dd)

res = np.load('simulated_dataset.npy', allow_pickle=True)
res = res[()]
print(type(res))
Ws = t.ones(len(dataset))
ewg = CDTools.tools.initializers.exit_wave_geometry
probe_basis, probe_shape, det_slice = ewg(res['basis'],
                                          dataset[0][1].shape,
                                          dataset.wavelength,
                                          dataset.detector_geometry['distance'],
                                          center=None,
                                          padding=0)
print('object', res['obj'].shape)
obj = res['obj'][..., 200:275, 200:275]
# print('obj', obj)
a = obj[0, 0, :, :]
b = obj[0, 1, :, :]
c = obj[1, 0, :, :]
d = obj[1, 1, :, :]
for i in [a, b, c, d]:
    plt.imshow(np.real(i))
    # plt.imshow(np.log(np))
    plt.colorbar()
    plt.show()
print(np.allclose(np.real(a), np.real(b)))
print('a', a[..., 10:20, 10:20])
print('b', b[..., 10:20, 10:20])
models = [CDTools.models.FancyPtycho(dataset.wavelength, dataset.detector_geometry, probe_basis,
                                     res['probe'], component, surface_normal=t.tensor([0., 0., 1.], dtype=t.float32),
                                     min_translation=t.tensor([0, 0], dtype=t.float32),
                                     background=t.tensor(res['background']), translation_offsets=None, mask=None,
                                     weights=Ws, translation_scale=1, saturation=None,
                                     probe_support=None, oversampling=1,
                                     loss='amplitude mse', units='um') for component in [a, b, c, d]]
# print('model is created')
# for model in models:
#     model.inspect()
#     plt.show()

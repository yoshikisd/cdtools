from __future__ import division, print_function, absolute_import
import numpy as np
import torch as t
from matplotlib import pyplot as plt
import pickle
import argparse

from CDTools.tools import cmath, plotting
from CDTools.tools.analysis import *


def make_argparser():
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('file', help='The reconstruction file to calculate metrics for')
    parser.add_argument('--use-probe', '-up', action='store_true', help='Use the probe instead of the object to align the reconstructions')
    return parser



if __name__ == '__main__':

    args = make_argparser().parse_args()

    with open(args.file, 'rb') as f:
        dataset = pickle.load(f)

    # This converts from a list of dictionaries to a dictionary of lists
    # It's safe to assume that all elements have the same set of keys
    if type(dataset) == type([]):
        dataset = {key: [element[key] for element in dataset]
                   for key in dataset[0]}
        calc_prtf = True
    else:
        # If it's a length-one reconstruction
        dataset = {key: [dataset[key]] for key in dataset}
        calc_prtf = False

    # Orthogonalize the probes
    dataset['probe'] = [orthogonalize_probes(p) for p in dataset['probe']]
    
    print('hi')

    synth_probe, synth_obj, aligned_objs = synthesize_reconstructions(
        dataset['probe'], dataset['obj'], args.use_probe)
    print('hey')
    if calc_prtf:
        freqs, prtf = calc_consistency_prtf(synth_obj, aligned_objs, dataset['basis'][0], nbins=200)

    # Either plot the only probe, or plot the dominant probe
    if len(synth_probe.shape) == 2:
        plotting.plot_phase(synth_probe,basis=dataset['basis'][0])
        plotting.plot_amplitude(synth_probe,basis=dataset['basis'][0])
        plotting.plot_colorized(synth_probe,basis=dataset['basis'][0])
    else:
        # Plot as many probes as exist
        try:
            for i in range(0,50):
                plotting.plot_phase(synth_probe[i],basis=dataset['basis'][0])
                plt.title('Probe ' + str(i+1) + 'Phase')
                plotting.plot_amplitude(synth_probe[i],basis=dataset['basis'][0])
                plt.title('Probe ' + str(i+1) + ' Amplitude')
                plotting.plot_colorized(synth_probe[i],basis=dataset['basis'][0])
                plt.title('Probe ' + str(i+1) + ' Colorized')
        except IndexError:
            pass
      
    plotting.plot_amplitude(aligned_objs[0][300:-300,300:-300],basis=dataset['basis'][0])
    plotting.plot_colorized(aligned_objs[0][300:-300,300:-300],basis=dataset['basis'][0])
    plotting.plot_phase(aligned_objs[0][300:-300,300:-300],basis=dataset['basis'][0])

    plotting.plot_amplitude(synth_obj[300:-300,300:-300],basis=dataset['basis'][0])
    plotting.plot_colorized(synth_obj[300:-300,300:-300],basis=dataset['basis'][0])
    plotting.plot_phase(synth_obj[300:-300,300:-300],basis=dataset['basis'][0])

    
    try:
        real_translations = dataset['translation'][0]
        real_translations -= np.min(real_translations,axis=0)[None,:]
        real_translations = real_translations
        plotting.plot_translations(real_translations)
        
        plotting.plot_nanomap(real_translations,dataset['weights'][0])
    except:
        pass

    plt.figure()
    plt.imshow(np.sqrt(dataset['background'][0]))

    if calc_prtf:
        plt.figure()
        plt.plot(freqs*1e-6, prtf)
        plt.xlabel('Spatial Frequency (cycles/um)')
        plt.ylabel('Consistency Based PRTF')
        plt.grid()

    plt.show()

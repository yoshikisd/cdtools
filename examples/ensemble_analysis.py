from __future__ import division, print_function, absolute_import

import numpy as np
from matplotlib import pyplot as plt
import pickle

from CDTools.tools import cmath, plotting
from CDTools.tools.analysis import *


#
# Note that much of this functionality is duplicated by the convenience
# script. Try running:
#
# python -m CDTools.scripts.synthesize example_reconstructions/gold_balls_ensemble.pickle
#
# Which will perform much of the same analysis on any saved reconstruction
# ensemble
#


with open('example_reconstructions/gold_balls_ensemble.pickle', 'rb') as f:
    dataset = pickle.load(f)

# This converts from a list of dictionaries to a dictionary of lists
# It's safe to assume that all elements have the same set of keys
if type(dataset) == type([]):
    dataset = {key: [element[key] for element in dataset]
               for key in dataset[0]}


# Now we synthesize the object using the tool from CDTools
synth_probe, synth_obj, aligned_objs = synthesize_reconstructions(
    dataset['probe'], dataset['obj'])

# And then we calculate the consistency PRTF from this
freqs, prtf = calc_consistency_prtf(synth_obj, aligned_objs, dataset['basis'][0])

# Plot the first mode in detail
plotting.plot_phase(synth_probe[0],basis=dataset['basis'][0])
plotting.plot_amplitude(synth_probe[0],basis=dataset['basis'][0])
plotting.plot_colorized(synth_probe[0],basis=dataset['basis'][0])

# Just plot the colorized version of the subdominant modes
plotting.plot_colorized(synth_probe[1],basis=dataset['basis'][0])
plotting.plot_colorized(synth_probe[2],basis=dataset['basis'][0])
      
plotting.plot_amplitude(synth_obj,basis=dataset['basis'][0])
plotting.plot_colorized(synth_obj,basis=dataset['basis'][0])
plotting.plot_phase(synth_obj,basis=dataset['basis'][0])
    
# Now plot the PRTF
plt.figure()
plt.plot(freqs*1e-6, prtf)
plt.xlabel('Spatial Frequency (cycles/um)')
plt.ylabel('Consistency Based PRTF')

plt.show()

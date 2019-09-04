from __future__ import division, print_function, absolute_import

import CDTools
import pickle

# Load the data
filename = 'example_data/AuBalls_700ms_30nmStep_3_6SS_filter.cxi'
dataset = CDTools.datasets.Ptycho_2D_Dataset.from_cxi(filename)

results = []

for idx in range(25):
    print('Starting Reconstruction', idx)
    
    # Create a new model each time
    model = CDTools.models.FancyPtycho.from_dataset(dataset,n_modes=3,
                                                    randomize_ang=0.1*np.pi)

    # Work on the GPU
    model.to(device='cuda')
    dataset.get_as(device='cuda')

    # Run the reconstruction
    for i, loss in enumerate(model.Adam_optimize(30, dataset, batch_size=100)):
        print(i,loss)

    # And add the results to the ensemble
    results.append(model.save_results(dataset))

    
# Save out the ensemble
with open('example_reconstructions/gold_balls_ensemble.pickle', 'wb') as f:
    pickle.dump(results,f)
    

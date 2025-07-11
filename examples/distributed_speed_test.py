'''This is a testing script to study how the reconstruction speed
and convergence rate scales with the number of GPUs utilized.

The test is set up so that you can run n-trials for each number of GPUs
you want to study and plot statistics of loss-versus-time as a function
of GPU counts. 

This test is based on fancy_ptycho_multi_gpu_ddp.py and fancy_ptycho.py.

'''

from cdtools.tools import distributed as dist
import os

# If you're running on AMD CPUs, you need to include this or else you will get a
# threading layer error. 
os.environ['MKL_THREADING_LAYER'] = 'GNU'

# This will execute the multi_gpu_reconstruct upon running this file
if __name__ == '__main__':
    # Define the number of GPUs to use.
    world_sizes = [1+i for i in range(7)] 

    # How many reconstruction runs to perform for statistics
    runs = 3

    # Define where the single-GPU script is located
    #script_path = 'fancy_ptycho_speed_test.py'
    script_path = 'gold_ball_ptycho.py'

    # Define where the loss-vs-time data is being stored in
    output_dir = 'example_loss_data4'

    # Define what prefix you want on the file
    file_prefix = 'speed_test'

    # Run the test
    dist.run_speed_test(world_sizes, runs, script_path, output_dir, file_prefix)
    
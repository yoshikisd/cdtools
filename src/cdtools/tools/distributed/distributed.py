"""Contains functions to make reconstruction scripts compatible
with multi-GPU distributive approaches in PyTorch.

Multi-GPU computing here is based on distributed data parallelism, where
each GPU is given identical copies of a model and performs optimization
using different parts of the dataset. After the parameter gradients
are calculated (`loss.backwards()`) on each GPU, the gradients need to be
synchronized and averaged across all participating GPUs. 

The functions in this module assist with gradient synchronization,
setting up conditions necessary to perform distributive computing, and
executing multi-GPU jobs. 
"""

import torch as t
import torch.distributed as dist
import datetime
import os
import subprocess
import argparse
import runpy
from ast import literal_eval
from matplotlib import pyplot as plt
import pickle
import numpy as np
from typing import Callable
from pathlib import Path
from cdtools.models import CDIModel

DISTRIBUTED_PATH = os.path.dirname(os.path.abspath(__file__))
MIN_INT64 = np.iinfo(np.int64).min
MAX_INT64 = np.iinfo(np.int64).max

__all__ = ['sync_and_avg_gradients', 
           'run_single_to_multi_gpu',
           'run_single_gpu_script',
           'report_speed_test',
           'run_speed_test']


def sync_and_avg_gradients(model: CDIModel):
    """
    Synchronizes the average of the model parameter gradients across all
    participating GPUs.

    Parameters:
        model: CDIModel
            Model for CDI/ptychography reconstruction  
    """
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM) 
            param.grad.data /= model.world_size


def run_single_gpu_script(script_path: str,
                          backend: str = 'nccl',
                          timeout: int = 30,
                          nccl_p2p_disable: bool = True,
                          seed: int = None):
    """
    Wraps single-GPU reconstruction scripts to be ran as a multi-GPU job via
    torchrun calls. 

    `cdtools.tools.distributed.run_single_gpu_script` is intended to be called in a script
    (e.g., cdtools.tools.distributed.single_to_multi_gpu) with the following form:

    ```
    # multi_gpu_job.py
    import cdtools.tools.distributed as dist
    if __name__ == '__main__':
        dist.run_single_to_multi_gpu(script_path='YOUR_RECONSTRUCTION_SCRIPT.py',
                                     backend='nccl',
                                     timeout=30,
                                     nccl_p2p_disable=True)
    ```
    
    `torchrun` should then be used to run this script as a distributive job using,
    for instance:
    
    ```
    torchrun --nnodes=1 --nproc_per_node=<number of GPUs> multi_gpu_job.py
    ```

    `torchrun` will spawn a number of subprocesses equal to the number of GPUs specified 
    (--nproc_per_node). On each subprocess, `cdtools.tools.distributed.run_single_gpu_script`
    will set up process groups (lets each GPU communicate with each other) and environment
    variables necessary for multi-GPU jobs. The single-GPU script will then be ran by
    each subprocess, where gradient synchronization will be faciliated by
    `cdtools.tools.distributed.sync_and_avg_gradients` calls from `cdtools.Reconstructors`
    while data shuffling/loading is handled by `cdtools.Reconstructor.setup_dataloader`.

    
    If you want to use specific GPU IDs for reconstructions, you need to set up
    the environment variable `CDTOOLS_GPU_IDS` rather than `CUDA_VISIBLE_DEVICES`. 
    If you wanted to use GPU IDs `1, 3, 4` for example, write:
    
    ```
    CDTOOLS_GPU_IDS=1,3,4 torchrun --nnodes=1 --nproc_per_node=<number of GPUs> multi_gpu_job.py
    ```
    
    NOTE: For each subprocess `cdt-torchrun` creates, the environment variable
          `CUDA_VISIBLE_DEVICES` will be (re)defined as the GPU rank.

    Parameters:
        script_name: str
            The file path of the single-GPU script (either full or relative).
            If you're using a relative path, make sure the string doesn't start
            with a backslash.
        backend: str
            Multi-gpu communication backend to use. Default is the 'nccl' backend,
            which is the only supported backend for CDTools.
            See https://pytorch.org/docs/stable/distributed.html for additional info
            about PyTorch-supported backends.
        timeout: int
            Timeout for operations executed against the process group in seconds. 
            Default is 30 seconds. After timeout has been reached, all subprocesses
            will be aborted and the process calling this method will crash. 
        nccl_p2p_disable: bool
            Disable NCCL peer-2-peer communication. If you find that all your GPUs
            are at 100% useage but the program isn't doing anything, try enabling
            this variable.
        seed: int
            Seed for generating random numbers.
            
    """
    
    # Check if the file path actually exists before starting the process group
    if not os.path.exists(script_path):
        raise FileNotFoundError(f'Cannot open file: {os.path.join(os.getcwd(), script_path)}')
    
    # Enable/disable NCCL peer-to-peer communication. The boolean needs to be converted into
    # a string for the environment variable.
    os.environ['NCCL_P2P_DISABLE'] = str(int(nccl_p2p_disable))
    
    ##########   Force each subprocess to see only the GPU ID we assign it  ###########
    # Why am I doing this? If this constraint is not imposed, then calling all_reduce will
    # cause all subprocess Ranks to occupy memory on both their own respective GPUs (normal) 
    # as well as Rank 0's GPU (not intended behavior). The root cause is not entirely clear 
    # but there are two ways to avoid this empirically:
    #   1) Force each subprocesses' CUDA_VISIBLE_DEVICE to be their assigned GPU ids.
    #   2) Within the reconstruction script, change `device='cuda'` to `device=f'cuda{model.rank}'`

    # The GPU rank and world_size is visible as an environment variable through torchrun calls.
    rank = int(os.environ.get('RANK'))
    world_size = int(os.environ.get('WORLD_SIZE'))

    # If the CDTOOLS_GPU_IDS environment variable is defined, then assign based
    # on the GPU IDS provided in that list. Otherwise, use the rank for the GPU ID.
    gpu_ids = os.environ.get('CDTOOLS_GPU_IDS')
    
    if gpu_ids is None:
        gpu_id = rank
    else:
        gpu_id = literal_eval(gpu_ids)[rank]

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    ################################  Run the script  #################################

    if rank == 0:
        print(f'[INFO]: Starting up multi-GPU reconstructions with {world_size} GPUs.')

    # Start up the process group (lets the different subprocesses talk with each other)
    dist.init_process_group(backend=backend, timeout=datetime.timedelta(seconds=timeout))

    try:     
        # Force all subprocesses to either use the pre-specified or Rank 0's RNG seed
        if seed is None:
            seed_local = t.tensor(np.random.randint(MIN_INT64, MAX_INT64), device='cuda', dtype=t.int64)
            dist.broadcast(seed_local, 0)
            seed = seed_local.item()

        t.manual_seed(seed)

        # Run the single-GPU reconstruction script 
        runpy.run_path(script_path, run_name='__main__')

        # Let the user know the job is done
        if rank == 0:
            print(f'[INFO]: Reconstructions complete. Terminating process group.')

    finally:
        # Kill the process group
        dist.destroy_process_group()   
        if rank == 0:
            print(f'[INFO]: Process group terminated. Multi-GPU job complete.')


def run_single_to_multi_gpu():
    """
    Runs a single-GPU reconstruction script as a single-node multi-GPU job via torchrun.
    
    This function can be executed as a python console script as `cdt-torchrun` and
    serves as a wrapper over a `torchrun` call to `cdtools.tools.distributed.single_to_multi_gpu`.
    
    In the simplest case, a single-GPU script can be ran as a multi-GPU job using
    the following `cdt-torchrun` call in the command line
    ```
    cdt-torchrun --ngpus=<number of GPUs> YOUR_RECONSTRUCTION_SCRIPT.py
    ```
    
    which is equivalent to the following `torchrun` call
    ```
    torchrun 
        --standalone 
        --nnodes=1 
        --nproc_per_node=<number of GPUs> 
        -m cdtools.tools.distributed.single_to_multi_gpu
        --backend='nccl'
        --timeout=30
        --nccl_p2p_disable=1
        YOUR_RECONSTRUCTION_SCRIPT.py
    ```
    
    With a single node (--nnodes=1), `cdt-torchrun` will launch a given number of subprocesses 
    equivalent to the number of GPUs specified. This number must be less than or equal to the
    actual number of GPUs available on your node.

    If you want to use specific GPU IDs for reconstructions, you need to set up
    the environment variable `CDTOOLS_GPU_IDS` rather than `CUDA_VISIBLE_DEVICES`. 
    If you wanted to use GPU IDs `1, 3, 4` for example, write:
    
    ```
    CDTOOLS_GPU_IDS=1,3,4 cdt-torchrun --ngpus=3 YOUR_RECONSTRUCTION_SCRIPT.py
    ```

    If additional `torchrun` arguments need to be passed, you may need to make a direct
    `torchrun` call rather than use `cdt-torchrun`. You may also submit an issue/PR.

    NOTE: `cdt-torchrun` has only been tested using the 'nccl' backend, NCCL peer-to-peer communication
          disabled, and using 1 node. 

    Arguments:
        script_path: str
            Path of the single-GPU script (either full or partial path).
        --ngpus: int
            Number of GPUs to use.
        --nnodes: int
            Optional, number of nodes. Default 1; more than 1 nodes has not been tested.
        --backend: str
            Optional, communication backend for distributed computing (either `nccl` or `gloo`).
            Default is `nccl`
        --timeout: int
            Optional, time in seconds before the distributed process is killed. 
            Default is 30 seconds.
        --nccl_p2p_disable: int
            Optional, disable (1) or enable (0) NCCL peer-to-peer communication. Default
            is 1.
        
    """
    # Define the arguments we need to pass to dist.script_wrapper
    parser = argparse.ArgumentParser()

    parser.add_argument('--ngpus',
                        type=int,
                        help='Number of GPUs to use (called --nproc_per_node in torchrun)')
    parser.add_argument('--nnodes', 
                        type=str, 
                        default=1,
                        help='Number of nodes participating in distributive computing.')
    parser.add_argument('--backend', 
                        type=str, 
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='Communication backend (nccl or gloo)')
    parser.add_argument('--timeout', 
                        type=int, 
                        default=30,
                        help='Time before process is killed in seconds')
    parser.add_argument('--nccl_p2p_disable', 
                        type=int, 
                        default=1,
                        choices=[0,1],
                        help='Disable (1) or enable (0) NCCL peer-to-peer communication')
    parser.add_argument('script_path', 
                        type=str, 
                        help='Single GPU script file name (with or without .py extension)')
    
    # Get the arguments
    args = parser.parse_args()
    
    # Perform the torchrun call of the wrapped function
    subprocess.run(['torchrun', # We set up the torchrun arguments first
                    '--standalone', # Indicates that we're running a single machine, multiple GPU job.
                    f'--nnodes={args.nnodes}', 
                    f'--nproc_per_node={args.ngpus}', 
                    '-m',
                    'cdtools.tools.distributed.single_to_multi_gpu', 
                    f'--backend={args.backend}',
                    f'--timeout={args.timeout}',
                    f'--nccl_p2p_disable={args.nccl_p2p_disable}',
                    f'{args.script_path}'])
    

def report_speed_test(func: Callable):
    """
    Decorator function which saves the loss-versus-time/epoch history of a 
    function-wrapped reconstruction script to a pickle dump file. This function
    is intended to be used for multi-GPU test studies performed with
    `cdtools.tools.distributed.run_speed_test`, which sets several environment
    variables specifing the name and directory of the result files to be saved.
    
    If the directory specified by `CDTOOLS_SPEED_TEST_RESULT_DIR` does not exist,
    one will be created in the current directory.

    Parameters:
        func: Callable
            The entire reconstruction script wrapped in a function. Within the
            script, the function must be called with an if-name-main statement.
            Additionally, the function must return the reconstructed model.
    """
    def wrapper():
        # Figure out how to name the save file and where to save it to
        # These environment variables are provided by run_speed_test
        trial_number = int(os.environ.get('CDTOOLS_TRIAL_NUMBER'))
        save_dir = os.environ.get('CDTOOLS_SPEED_TEST_RESULT_DIR')
        file_prefix = os.environ.get('CDTOOLS_SPEED_TEST_PREFIX')

        # Check if the save path is valid
        # Make sure the directory exists; or else create it
        Path(save_dir).mkdir(parents=False, exist_ok=True)

        # Run the script
        model = func()

        # Save the model and loss history, but only using the rank 0 process
        if model.rank == 0:
            # Set up the file name:
            file_name = f'{file_prefix}_nGPUs_{model.world_size}_TRIAL_{trial_number}.pkl'
            # Grab the loss and time history
            loss_history = model.loss_history
            time_history = model.loss_times
            # Store quantities in a dictionary
            dict = {'loss history':loss_history,
                    'time history':time_history,
                    'nGPUs':model.world_size,
                    'trial':trial_number}
            
            # Save the quantities
            with open (os.path.join(save_dir, file_name), 'wb') as save_file:
                pickle.dump(dict, save_file)
            
            print(f'[INFO]: Saved results to: {file_name}') 
    return wrapper


def run_speed_test(world_sizes: int, 
                   runs: int,
                   script_path: str,
                   output_dir: str,
                   file_prefix: str = 'speed_test'):
    """
    Executes a reconstruction script `n` x `m` times using `n` GPUs and `m` trials 
    per GPU count using cdt-torchrun.

    If the directory specified by `output_dir` does not exist,
    one will be created in the current directory.

    Parameters:
        world_sizes: list[int]
            Number of GPUs to use. User can specify several GPU counts in a list.
        runs: int
            How many repeat reconstructions to perform
        script_path: str
            Path of the single-gpu reconstruction script.
        output_dir: str
            Directory of the loss-vs-time/epoch data generated for the speed test.
        file_prefix: str
            Prefix of the speed test result file names
    """
    # Set stuff up for plots
    fig, (ax1,ax2,ax3) = plt.subplots(1,3)

    # Store the value of the single GPU time
    time_1gpu = 0
    std_1gpu = 0

    for world_size in world_sizes:
        # Make a list to store the values
        time_list = []
        loss_hist_list = []

        for i in range(runs):
            print(f'[INFO]: Resetting the model...')
            print(f'[INFO]: Starting run {i+1}/{runs} on {world_size} GPU(s)')

            # The scripts running speed tests need to read the trial number
            # they are on using an environment variable
            os.environ['CDTOOLS_TRIAL_NUMBER'] = str(i)
            os.environ['CDTOOLS_SPEED_TEST_RESULT_DIR'] = output_dir
            os.environ['CDTOOLS_SPEED_TEST_PREFIX'] = file_prefix
            
            # Run cdt-torchrun
            try:
                subprocess.run(['cdt-torchrun',
                                f'--ngpus={world_size}',
                                f'{script_path}'],
                                check=True)
            except subprocess.CalledProcessError as e:
                print(e)

            print(f'[INFO]: Reconstruction complete. Loading loss results...')

            with open(os.path.join(output_dir, f'{file_prefix}_nGPUs_{world_size}_TRIAL_{i}.pkl'), 'rb') as f:
                results = pickle.load(f)
            time_list.append(results['time history'])
            loss_hist_list.append(results['loss history'])

            
        # Calculate the statistics
        time_mean = np.array(time_list).mean(axis=0)/60
        time_std = np.array(time_list).std(axis=0)/60
        loss_mean = np.array(loss_hist_list).mean(axis=0)
        loss_std = np.array(loss_hist_list).std(axis=0)

        # If a single GPU is used, store the time
        if world_size == 1:
            time_1gpu = time_mean[-1]
            std_1gpu = time_std[-1]

        # Calculate the speed-up relative to using a single GPU
        speed_up_mean = time_1gpu / time_mean[-1] 
        speed_up_std = speed_up_mean * \
            np.sqrt((std_1gpu/time_1gpu)**2 + (time_std[-1]/time_mean[-1])**2)

        # Add another plot
        ax1.errorbar(time_mean, loss_mean, yerr=loss_std, xerr=time_std,
                    label=f'{world_size} GPUs')
        ax2.errorbar(np.arange(0,loss_mean.shape[0]), loss_mean, yerr=loss_std,
                    label=f'{world_size} GPUs')
        ax3.errorbar(world_size, speed_up_mean, yerr=speed_up_std, fmt='o')
        
    # Plot
    fig.suptitle(f'Multi-GPU performance test | {runs} runs performed')
    ax1.set_yscale('log')
    ax1.set_xscale('linear')
    ax2.set_yscale('log')
    ax2.set_xscale('linear')
    ax3.set_yscale('linear')
    ax3.set_xscale('linear')
    ax1.legend()
    ax2.legend()
    ax1.set_xlabel('Time (min)')
    ax1.set_ylabel('Loss')
    ax2.set_xlabel('Epochs')
    ax3.set_xlabel('Number of GPUs')
    ax3.set_ylabel('Speed-up relative to single GPU')
    plt.show()
    print(f'[INFO]: Multi-GPU speed test completed.')
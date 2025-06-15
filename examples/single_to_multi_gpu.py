"""
A wrapper script intended to run single-GPU scripts as
a multi-GPU job

This script is intended to be called by torchrun. It is set
up so that the group process handling (init and destroy) are
handled here. The core of the function calls the reconstruction
script of interest.

Currently, this is the torchrun command I'm calling to use
4 GPUs:

torchrun --nnodes=1 --nproc_per_node=4 single_to_multi_gpu.py
"""
import os
import datetime
import torch.distributed as dist
import torch as t



if __name__ == '__main__':
    # If this script is called by torchrun, several environment
    # variables are created that we need to store as variables
    rank = int(os.environ.get('RANK'))
    world_size = int(os.environ.get('WORLD_SIZE'))

    os.environ['NCCL_P2P_DISABLE'] = str(int(True))
    t.cuda.set_device(rank)
    timeout = datetime.timedelta(seconds=300)

    # Start up the process group (needed so the different
    # subprocesses can talk with each other)
    dist.init_process_group(backend='nccl',
                            timeout=timeout)
    
    try:     
        # Run the single-GPU reconstruction script
        import fancy_ptycho_torchrun

    finally:
        # Kill the process group
        dist.destroy_process_group()
import cdtools
from matplotlib import pyplot as plt

# To use multiple GPUs, we need to import a few additional packages
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier
import torch.multiprocessing as mp
import os

# While not strictly necessary, it's super useful to have in the event
# the computation hangs by defining a timeout period. 
import datetime 
timeout = datetime.timedelta(seconds=60)   # Terminate if things hang for 60s.

# We will need to specify what multiprocessing backend we want to use.
# PyTorch supports a few backends (gloo, MPI, NCCL). We will use NCCL, or
# NVIDIA Collective Communications Library, as it's the fastest one.
#
# It's also the only one that works with the current multi-GPU implementation...
BACKEND = 'nccl'

# We need to wrap the script inside a function in order to use "mp.spawn"
# which will help distribute the work to multiple GPUs
#
# In fancier terms, we will use mp.spawn to create several processes
# that will work on the model using N-number of GPUs, (a.k.a., 'WORLD_SIZE')
# Each process will be given to one GPU that's assigned a number called 
# a RANK (which ranges from 0 to WORLD_SIZE-1).
def multi_gpu_reconstruct(rank: int, 
                          world_size: int):
    """Perform the reconstruction using several GPUs
    Parameters:
        rank: int
            The rank of the GPU to be used. Value should be within
            [0, world_size-1]

        world_size: int
            The total number of GPUs to use
    """
    # We need to initialize the distributed process group
    # before calling any other method
    init_process_group(backend=BACKEND,
                       rank=rank,
                       world_size=world_size,
                       timeout=timeout)
    
    filename = 'example_data/lab_ptycho_data.cxi'
    dataset = cdtools.datasets.Ptycho2DDataset.from_cxi(filename)

    model = cdtools.models.FancyPtycho.from_dataset(
        dataset,
        n_modes=3,
        oversampling=2, 
        probe_support_radius=120, 
        propagation_distance=5e-3,
        units='mm', 
        obj_view_crop=-50,
    )

    # We need to adjust the device string to also indicate which GPU this
    # process is using
    device = f'cuda:{rank}'
    model.to(device=device)
    dataset.get_as(device=device)

    # We now wrap the model with DistributedDataParallel (DDP), which allows
    # data parallelism by synchronizing gradients across each copy of the
    # model in the different GPUs.
    model = DDP(model,
                device_ids=[rank],  # Tells DDP which GPU the model lives in
                output_device=rank, # Tells DDP which GPU to output to
                find_unused_parameters=True) # TODO: Understand what this is really doing...

    # As a sanity check, we wait for all GPUs to catch up to barrier() before
    # running optimization
    barrier()
    

    # Since our model is now wrapped in DDP, all CDTools methods have to be
    # called using 'model.module' rather than just 'model'.
    #
    # We also need to pass the rank and world_size to Adam_optimize
    for loss in model.module.Adam_optimize(50, 
                                           dataset, 
                                           lr=0.02, 
                                           batch_size=10,
                                           rank=rank,
                                           num_workers=world_size):
        
        # We can still perform model.inspect and model.report, but we want
        # to only let 1 GPU handle plotting/printing rather than get N copies
        # from all N GPUs.
        if rank == 0:
            print(model.module.report())
        
        # We set up the model.inspect this way to only let GPU 0 plot and
        # prevent the other GPUs from running far ahead of GPU 0, which
        # seems to cause bugs (GPU processes dissapear from nvidia-smi)
        if model.module.epoch % 10 == 0:
            if rank == 0:
                model.module.inspect(dataset)
            barrier()


    # We set up another barrier to make sure all GPUs catch up before
    # starting another reconstruction loop
    barrier()

    for loss in model.module.Adam_optimize(50, 
                                           dataset,  
                                           lr=0.005, 
                                           batch_size=50,
                                           rank=rank,
                                           num_workers=world_size):
        if rank == 0:
            print(model.module.report())
        
        if model.epoch % 10 == 0:
            if rank == 0:
                model.module.inspect(dataset)
            barrier()

    # Again, set up another barrier to let all GPUs catch up
    barrier()
    
    model.module.tidy_probes() # TODO: Check how the multi-GPU implementation handles tidying probes.

    # Only let one GPU handle plotting stuff.
    if rank == 0:
        model.module.inspect(dataset)
        model.module.compare(dataset)
        plt.show()

# This will execute the multi_gpu_reconstruct upon running this file
if __name__ == '__main__':
    # We need to add some stuff to the enviromnent 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'

    # Define the number of GPUs to use.
    world_size = 4

    # Write a try/except statement to help the subprocesses (and GPUs)
    # terminate gracefully. Otherwise, you may have stuff loaded on
    # several GPU even after terminating.
    try:
        # Spawn the processes
        mp.spawn(multi_gpu_reconstruct,
                args=(world_size,),
                nprocs=world_size,
                join=True)
        
        # Always destroy the process group when you're done
        destroy_process_group()

    except Exception as e:
        # If something breaks, we try to make sure that the
        # process group is destroyed before the program fully
        # terminates
        print(e)
        destroy_process_group()
    

"""This module contains the base Reconstructor class for performing
optimization ('reconstructions') on ptychographic/CDI models.

The Reconstructor class is designed to resemble so-called
'Trainer' classes that (in the language of the AI/ML folks) that
handles the 'training' of a model given some dataset and optimizer.

The subclasses of the Reconstructor class are required to implement
their own data loaders and optimizer adjusters
"""

import torch as t
from torch.utils import data as torchdata
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import threading
import queue
import time
from contextlib import contextmanager
from cdtools.tools.data import nested_dict_to_h5, h5_to_nested_dict, nested_dict_to_numpy, nested_dict_to_torch
from cdtools.datasets.ptycho_2d_dataset import Ptycho2DDataset
from cdtools.models import CDIModel
import torch.distributed as dist
from typing import Tuple, List

__all__ = ['Reconstructor']

class Reconstructor:
    def __init__(self,
                 model: CDIModel,
                 dataset: Ptycho2DDataset,
                 subset: List[int] = None,
                 thread: bool = True):
        
        # Store parameters as attributes
        self.multi_gpu_used = model.multi_gpu_used
        self.world_size = model.world_size
        self.rank = model.rank
        self.subset = subset
        self.thread = thread

        # Initialize some attributes that must be defined by other methods
        self.optimizer = None
        self.scheduler = None
        self.data_loader = None
        #self.epoch = model.epoch
        
        # Store either the original or DDP-wrapped model, along with
        # references to model attributes/methods
        """ For now, don't do DDP-wrapping; check if porting the bits
        and pieces from CDIModel works before doing DDP.
        if self.multi_gpu_used:
            self.model = DDP(model, 
                             device_ids=[0]) # This is used if CUDA_VISIBLE_DEVICES is manually set
            store_references(self.model.module)
        
        else:
        """
        self.model = model
        #store_model_attributes(self.model)

        # Store the dataset
        if subset is not None:
            # if subset is just one pattern, turn into a list for convenience
            if type(subset) == type(1):
                subset = [subset]
            dataset = torchdata.Subset(dataset, subset)
        
        self.dataset = dataset

        
    def setup_dataloader(self, **kwargs):
        """Sets up the dataloader 

        The dataloader needs to be defined manually for each subclass.
        While each subclass will likely use similar calls to 
        """
        raise NotImplementedError()


    def adjust_optimizer(self, **kwargs):
        """This is to allow us to set up parameters for whatever optimizer we're
        interested in using.

        The different optimization schemes (Adam, LBFGS, SGD) seem to take in 
        different hyperparameters. This function is intended to modify the
        parameters

        This is not defined here. For each optimizer, the keyword
        arguments should be manually defined as parameters
        """
        raise NotImplementedError()


    def optimize(self,
                 iterations,
                 regularization_factor=None,
                 thread=True,
                 calculation_width=10):
        """Runs a round of reconstruction using the provided optimizer
        
        Formerly CDIModel.AD_optimize

        This is the basic automatic differentiation reconstruction tool
        which all the other, algorithm-specific tools, use. It is a
        generator which yields the average loss each epoch, ending after
        the specified number of iterations.

        By default, the computation will be run in a separate thread. This
        is done to enable live plotting with matplotlib during a reconstruction.
        If the computation was done in the main thread, this would freeze
        the plots. This behavior can be turned off by setting the keyword
        argument 'thread' to False.        

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run
        data_loader : torch.utils.data.DataLoader
            A data loader loading the CDataset to reconstruct
        optimizer : torch.optim.Optimizer
            The optimizer to run the reconstruction with
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Optional, a learning rate scheduler to use
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of parameters to pass the regularizer method
        thread : bool
            Default True, whether to run the computation in a separate thread to allow interaction with plots during computation
        calculation_width : int
            Default 10, how many translations to pass through at once for each round of gradient accumulation. This does not affect the result, but may affect the calculation speed.

        Yields
        ------
        loss : float
            The summed loss over the latest epoch, divided by the total diffraction pattern intensity
        """

        def run_epoch(stop_event=None):
            """Runs one full epoch of the reconstruction."""
            # If we're using DistributedSampler (likely the case if you're using 
            # multiple GPUs), we need to tell it which epoch we're on. Otherwise
            # data shuffling will not work properly
            if self.multi_gpu_used: 
                self.data_loader.sampler.set_epoch(self.model.epoch)

            # First, initialize some tracking variables
            normalization = 0
            loss = 0
            N = 0
            t0 = time.time()

            # The data loader is responsible for setting the minibatch
            # size, so each set is a minibatch
            for inputs, patterns in self.data_loader:
                normalization += t.sum(patterns).cpu().numpy()
                N += 1
                def closure():
                    self.optimizer.zero_grad()

                    # We further break up the minibatch into a set of chunks.
                    # This lets us use larger minibatches than can fit
                    # on the GPU at once, while still doing batch processing
                    # for efficiency
                    input_chunks = [[inp[i:i + calculation_width]
                                     for inp in inputs]
                                    for i in range(0, len(inputs[0]),
                                                   calculation_width)]
                    pattern_chunks = [patterns[i:i + calculation_width]
                                      for i in range(0, len(inputs[0]),
                                                     calculation_width)]

                    total_loss = 0
                    for inp, pats in zip(input_chunks, pattern_chunks):
                        # This check allows for graceful exit when threading
                        if stop_event is not None and stop_event.is_set():
                            exit()

                        # Run the simulation
                        sim_patterns = self.model.forward(*inp) 

                        # Calculate the loss
                        if hasattr(self, 'mask'):
                            loss = self.model.loss(pats,sim_patterns, mask=self.model.mask)
                        else:
                            loss = self.model.loss(pats,sim_patterns)

                        # And accumulate the gradients
                        loss.backward()

                        # For multi-GPU optimization, we need to average and
                        # sync the gradients + losses across all participating
                        # GPUs with an all-reduce call.
                        if self.multi_gpu_used:
                            for param in self.model.parameters():
                                if param.requires_grad:
                                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM) 
                                    param.grad.data /= self.model.world_size
                            
                            # Sum the loss value across all devices for reporting
                            dist.all_reduce(loss, op=dist.ReduceOp.SUM) 
                        
                        # Normalize the accumulating total loss by the number of GPUs used
                        total_loss += loss.detach() // self.model.world_size
                        

                    # If we have a regularizer, we can calculate it separately,
                    # and the gradients will add to the minibatch gradient
                    if regularization_factor is not None \
                       and hasattr(self.model, 'regularizer'):
                        loss = self.model.regularizer(regularization_factor)
                        loss.backward()

                        # For multi-GPU optimization, we need to average and
                        # sync the gradients + losses across all participating
                        # GPUs with an all-reduce call.
                        if self.multi_gpu_used:
                            for param in self.model.parameters():
                                if param.requires_grad:
                                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM) 
                                    param.grad.data /= self.model.world_size
                    
                    return total_loss

                # This takes the step for this minibatch
                loss += self.optimizer.step(closure).detach().cpu().numpy()
            
            loss /= normalization

            # We step the scheduler after the full epoch
            if self.scheduler is not None:
                self.scheduler.step(loss)

            self.model.loss_history.append(loss)
            self.model.epoch = len(self.model.loss_history)
            self.model.latest_iteration_time = time.time() - t0
            self.model.training_history += self.model.report() + '\n'
            return loss

        # We store the current optimizer as a model parameter so that
        # it can be saved and loaded for checkpointing
        self.current_optimizer = self.optimizer
        
        # If we don't want to run in a different thread, this is easy
        if not thread:
            for it in range(iterations):
                if self.model.skip_computation():
                    self.epoch = self.epoch + 1
                    if len(self.model.loss_history) >= 1:
                        yield self.model.loss_history[-1]
                    else:
                        yield float('nan')
                    continue

                yield run_epoch()
                    
                
        # But if we do want to thread, it's annoying:
        else:
            # Here we set up the communication with the computation thread
            result_queue = queue.Queue()
            stop_event = threading.Event()
            def target():
                try:
                    result_queue.put(run_epoch(stop_event))
                except Exception as e:
                    # If something bad happens, put the exception into the
                    # result queue
                    result_queue.put(e)

            # And this actually starts and monitors the thread
            for it in range(iterations):
                if self.model.skip_computation():
                    self.model.epoch = self.model.epoch + 1                    
                    if len(self.model.loss_history) >= 1:
                        yield self.model.loss_history[-1]
                    else:
                        yield float('nan')
                    continue

                calc = threading.Thread(target=target, name='calculator', daemon=True)
                try:
                    calc.start()
                    while calc.is_alive():
                        if hasattr(self.model, 'figs'):
                            self.model.figs[0].canvas.start_event_loop(0.01)
                        else:
                            calc.join()

                except KeyboardInterrupt as e:
                    stop_event.set()
                    print('\nAsking execution thread to stop cleanly - please be patient.')
                    calc.join()
                    raise e

                res = result_queue.get()

                # If something went wrong in the thead, we'll get an exception
                if isinstance(res, Exception):
                    raise res

                yield res

        # And finally, we unset the current optimizer:
        self.current_optimizer = None
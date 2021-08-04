"""This module contains the base CDIModel class for CDI Models.

The subclasses of the main CDIModel class are required to define their
own implementations of the following functions:

Loading and Saving
------------------
from_dataset
    Creates a CDIModel from an appropriate CDataset
simulate_to_dataset
    Creates a CDataset from the simulation defined in the model
save_results
    Saves out a dictionary with the recovered parameters
    

Simulation
----------
interaction
    Simulates exit waves from experimental parameters
forward_propagator
    The propagator from the experiment plane to the detector plane
backward_propagator
    Optional, the propagator from the detector plane to the experiment plane
measurement
    Simulates the detector readout from a detector plane wavefront
loss
    the loss function to report and use for automatic differentiation

"""

from __future__ import division, print_function, absolute_import

import torch as t
from torch.utils import data as torchdata
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import ticker
import numpy as np
import threading
import queue
import time
#import pytorch_warmup
from .complex_adam import MyAdam
from .complex_lbfgs import MyLBFGS
from matplotlib.backends.backend_pdf import PdfPages


__all__ = ['CDIModel']


class CDIModel(t.nn.Module):
    """This base model defines all the functions that must be exposed for a valid CDIModel subclass

    Most of the functions only raise a NotImplementedError at this level and
    must be explicitly defined by any subclass - these are noted explocitly
    in the module-level intro. The work of defining the various subclasses
    boils down to creating an appropriate implementation for this set of
    functions.
    """

    def __init__(self):
        super(CDIModel,self).__init__()
        self.iteration_count = 0
        
    def from_dataset(self, dataset):
        raise NotImplementedError()


    def interaction(self, *args):
        raise NotImplementedError()


    def forward_propagator(self, exit_wave):
        raise NotImplementedError()


    def backward_propagator(self, detector_wave):
        raise NotImplementedError()


    def measurement(self, detector_wave):
        raise NotImplementedError()


    def forward(self, *args):
        """The complete forward model
        
        This model relies on composing the interaction, forward propagator,
        and measurement functions which are required to be defined by all
        subclasses. It therefore should not be redefined by the subclasses.
        
        The arguments to this function, for any given subclass, will be
        the same as the arguments to the interaction function.
        """
        return self.measurement(self.forward_propagator(self.interaction(*args)))

    def loss(self, sim_data, real_data):
        raise NotImplementedError()


    def to(self, *args, **kwargs):
        super(CDIModel,self).to(*args,**kwargs)


    def simulate_to_dataset(self, args_list):
        raise NotImplementedError()
    
    def save_results(self):
        raise NotImplementedError()

    def AD_optimize(self, iterations, data_loader,  optimizer,\
                    scheduler=None, regularization_factor=None, thread=True,
                    calculation_width=10, warmup_scheduler=None):
        """Runs a round of reconstruction using the provided optimizer
        
        This is the basic automatic differentiation reconstruction tool
        which all the other, algorithm-specific tools, use.
        
        Like all the other optimization routines, it is defined as a
        generator function which yields the average loss each epoch.

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run
        dataset : CDataset
            The dataset to reconstruct against
        optimizer : torch.optim.Optimizer
            The optimizer to run the reconstruction with
        scheduler : torch.optim.lr_scheduler._LRScheduler
            Optional, a learning rate scheduler to use
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of parameters to pass the regularizer method
        thread : bool
            Default True, whether to run the computation in a separate thread to allow interaction with plots during computation
        calculation_width : int
            Default 10, how many translations to pass through at once for each round of gradient accumulation
        """
        # First, calculate the normalization
        normalization = 0
        for inputs, patterns in data_loader:
            normalization += t.sum(patterns).cpu().numpy()
            
        def run_iteration(stop_event=None):
            loss = 0
            N = 0
            t0 = time.time()
            for inputs, patterns in data_loader:
                N += 1
                def closure():
                    optimizer.zero_grad()
                    
                    input_chunks = [[inp[i:i + calculation_width]
                                     for inp in inputs]
                                    for i in range(0, len(inputs[0]),
                                                   calculation_width)]
                    pattern_chunks = [patterns[i:i + calculation_width]
                                      for i in range(0, len(inputs[0]),
                                                     calculation_width)]
                    
                    total_loss = 0
                    for inp, pats in zip(input_chunks, pattern_chunks):
                        # This is just used to allow graceful exit when threading
                        if stop_event is not None and stop_event.is_set():
                            exit()

                        sim_patterns = self.forward(*inp)
                        
                        #sim_patterns.retain_grad()
                        if hasattr(self, 'mask'):
                            loss = self.loss(pats,sim_patterns, mask=self.mask)
                        else:
                            loss = self.loss(pats,sim_patterns)

                        loss.backward()#retain_variables=True)
                        #plt.figure()
                        #plt.imshow(sim_patterns.grad.cpu()[0])
                        #plt.colorbar()
                        #plt.show()
                    
                        total_loss += loss.detach()    

                    #print('probe grad')
                    #print(t.mean(self.obj.grad))

                    if regularization_factor is not None \
                       and hasattr(self, 'regularizer'):
                        loss = self.regularizer(regularization_factor)
                        loss.backward()
                    return total_loss

                                    

                if warmup_scheduler is not None:
                    old_lrs = [group['lr'] for group in
                               optimizer.param_groups]
                    warmup_scheduler.dampen()
                    #print([group['lr'] for group in
                    #       optimizer.param_groups], end='\r')
                    
                loss += optimizer.step(closure).detach().cpu().numpy()

                if warmup_scheduler is not None:
                    for old_lr, group in zip(old_lrs, optimizer.param_groups):
                        group['lr'] = old_lr
    
                
            loss /= normalization
            if scheduler is not None:
                scheduler.step(loss)

            self.latest_loss = loss
            self.latest_iteration_time = time.time() - t0
            self.iteration_count += 1
            return loss

        if thread:
            result_queue = queue.Queue()
            stop_event = threading.Event()
            def target():
                result_queue.put(run_iteration(stop_event))
            
        for it in range(iterations):
            if thread:
                calc = threading.Thread(target=target, name='calculator', daemon=True)
                try:
                    calc.start()
                    while calc.is_alive():
                        if hasattr(self, 'figs'):
                            self.figs[0].canvas.start_event_loop(0.01) 
                        else:
                            calc.join()
                            
                except KeyboardInterrupt as e:
                    stop_event.set()
                    print('\nAsking execution thread to stop cleanly - please be patient.')
                    calc.join()
                    raise e
                    
                yield result_queue.get()
            else:
                yield run_iteration()


    def Adam_optimize(self, iterations, dataset, batch_size=15, lr=0.005,
                      schedule=False, amsgrad=False, subset=None,
                      regularization_factor=None, thread=True,
                      calculation_width=10, warmup=False):
        """Runs a round of reconstruction using the Adam optimizer
        
        This is generally accepted to be the most robust algorithm for use
        with ptychography. Like all the other optimization routines,
        it is defined as a generator function, which yields the average
        loss each epoch.

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run
        dataset : CDataset
            The dataset to reconstruct against
        batch_size : int
            Optional, the size of the minibatches to use
        lr : float
            Optional, The learning rate (alpha) to use
        schedule : float
            Optional, whether to use the ReduceLROnPlateau scheduler
        subset : list(int) or int
            Optional, a pattern index or list of pattern indices to use
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of parameters to pass the regularizer method
        thread : bool
            Default True, whether to run the computation in a separate thread to allow interaction with plots during computation
        calculation_width : int
            Default 1, how many translations to pass through at once for each round of gradient accumulation

        """

        if subset is not None:
            # if just one pattern, turn into a list for convenience
            if type(subset) == type(1):
                subset = [subset]
            dataset = torchdata.Subset(dataset, subset)
            
        # Make a dataloader
        data_loader = torchdata.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True)

        # Define the optimizer
        #optimizer = t.optim.Adam(self.parameters(), lr = lr, amsgrad=amsgrad)
        optimizer = MyAdam(self.parameters(), lr = lr, amsgrad=amsgrad)


        # Define the scheduler
        if schedule:
            scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2,threshold=1e-9)
        else:
            scheduler = None

        if warmup:
            print('Warmup is not currently implemented, sorry!')
            #warmup_scheduler = pytorch_warmup.UntunedLinearWarmup(optimizer)
        else:
            warmup_scheduler = None
            
        return self.AD_optimize(iterations, data_loader, optimizer,
                                scheduler=scheduler,
                                warmup_scheduler=warmup_scheduler,
                                regularization_factor=regularization_factor,
                                thread=thread,
                                calculation_width=calculation_width)


    def LBFGS_optimize(self, iterations, dataset, 
                       lr=0.1,history_size=2, subset=None,
                       regularization_factor=None, thread=True,
                       calculation_width=10):
        """Runs a round of reconstruction using the L-BFGS optimizer
        
        This algorithm is often less stable that Adam, however in certain
        situations or geometries it can be shockingly efficient. Like all
        the other optimization routines, it is defined as a generator
        function which yields the average loss each epoch.
        
        Note: There is no batch size, because it is a usually a bad idea to use
        LBFGS on anything but all the data at onece

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run
        dataset : CDataset
            The dataset to reconstruct against
        lr : float
            Optional, the learning rate to use
        history_size : int
            Optional, the length of the history to use.
        subset : list(int) or int
            Optional, a pattern index or list of pattern indices to ues
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of parameters to pass the regularizer method
        thread : bool
            Default True, whether to run the computation in a separate thread to allow interaction with plots during computation
        
        """
        if subset is not None:
            # if just one pattern, turn into a list for convenience
            if type(subset) == type(1):
                subset = [subset]
            dataset = torchdata.Subset(dataset, subset)
        
        # Make a dataloader. This basically does nothing but load all the
        # data at once
        data_loader = torchdata.DataLoader(dataset, batch_size=len(dataset))


        # Define the optimizer
        optimizer = t.optim.LBFGS(self.parameters(),
                                  lr = lr, history_size=history_size)
        #optimizer = MyLBFGS(self.parameters(),
        #                    lr = lr, history_size=history_size)
        
        return self.AD_optimize(iterations, data_loader, optimizer,
                                regularization_factor=regularization_factor,
                                thread=thread,
                                calculation_width=calculation_width)


    def SGD_optimize(self, iterations, dataset, batch_size=None,
                     lr=0.01, momentum=0, dampening=0, weight_decay=0,
                     nesterov=False, subset=None, regularization_factor=None,
                     thread=True, calculation_width=10):
        """Runs a round of reconstruction using the SGDoptimizer
        
        This algorithm is often less stable that Adam, but it is simpler
        and is the basic workhorse of gradience descent.

        Parameters
        ----------
        iterations : int
            How many epochs of the algorithm to run
        dataset : CDataset
            The dataset to reconstruct against
        batch_size : int
            Optional, the size of the minibatches to use
        lr : float
            Optional, the learning rate to use
        momentum : float
            Optional, the length of the history to use.
        subset : list(int) or int
            Optional, a pattern index or list of pattern indices to use
        regularization_factor : float or list(float)
            Optional, if the model has a regularizer defined, the set of parameters to pass the regularizer method
        thread : bool
            Default True, whether to run the computation in a separate thread to allow interaction with plots during computation
        calculation_width : int
            Default 1, how many translations to pass through at once for each round of gradient accumulation

        """

        if subset is not None:
            # if just one pattern, turn into a list for convenience
            if type(subset) == type(1):
                subset = [subset]
            dataset = torchdata.Subset(dataset, subset)
            
        # Make a dataloader
        if batch_size is not None:
            data_loader = torchdata.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True)
        else:
            data_loader = torchdata.DataLoader(dataset)


        # Define the optimizer
        optimizer = t.optim.SGD(self.parameters(),
                                lr = lr, momentum=momentum,
                                dampening=dampening,
                                weight_decay=weight_decay,
                                nesterov=nesterov)

        return self.AD_optimize(iterations, data_loader, optimizer,
                                regularization_factor=regularization_factor,
                                thread=thread,
                                calculation_width=calculation_width)


    def report(self):
        """Returns a string informing on the latest reconstruction iteration"""
        if hasattr(self, 'latest_loss'):
            return 'Iteration ' + str(self.iteration_count) + \
                  ' completed in %0.2f s with loss ' %\
                  self.latest_iteration_time + str(self.latest_loss)
        else:
            return 'No reconstruction iterations performed yet!'
        
    # By default, the plot_list is empty
    plot_list = []
    
    
    def inspect(self, dataset=None, update=True):
        """Plots all the plots defined in the model's plot_list attribute
  
        If update is set to True, it will update any previously plotted set
        of plots, if one exists, and then redraw them. Otherwise, it will
        plot a new set, and any subsequent updates will update the new set
        
        Optionally, a dataset can be passed, which then will plot any
        registered plots which need to incorporate some information from
        the dataset (such as geometry or a comparison with measured data).
        
        Plots can be registered in any subclass by defining the plot_list
        attribute. This should be a list of tuples in the following format:
        ( 'Plot Title', function_to_generate_plot(self), 
        function_to_determine_whether_to_plot(self))

        Where the third element in the tuple (a function that returns 
        True if the plot is relevant) is not required.

        Parameters
        ----------
        dataset : CDataset
            Optional, a dataset matched to the model type
        update : bool
            Whether to update existing plots or plot new ones
        
        """
                
        first_update = False
        if update and hasattr(self, 'figs') and self.figs:
            figs = self.figs
        elif update:
            figs = None
            self.figs = []
            first_update = True
        else:
            figs = None
            self.figs = []

        idx = 0
        for plots in self.plot_list:
            # If a conditional is included in the plot
            try:
                if len(plots) >=3 and not plots[2](self):
                    continue
            except TypeError as e:
                if len(plots) >= 3 and not plots[2](self, dataset):
                    continue

            name = plots[0]
            plotter = plots[1]

            if figs is None:
                fig = plt.figure()                
                self.figs.append(fig)
            else:
                fig = figs[idx]

            try:
                plotter(self,fig)
                plt.title(name)
                
            except TypeError as e:
                if dataset is not None:
                    try:
                        plotter(self, fig, dataset)
                        plt.title(name)
                        print('f', idx)
                        plt.savefig('img-{0}.pdf'.format(idx), bbox_inches='tight')
                    except (IndexError, KeyError, AttributeError, np.linalg.LinAlgError) as e:
                        pass

            except (IndexError, KeyError, AttributeError, np.linalg.LinAlgError) as e:
                pass

            idx += 1
            
            if update:
                plt.draw()
                fig.canvas.start_event_loop(0.001)
                
        if first_update:
            plt.pause(0.05 * len(self.figs))


    def compare(self, dataset):
        """Opens a tool for comparing simulated and measured diffraction patterns
        
        Parameters
        ----------
        dataset : CDataset
            A dataset containing the simulated diffraction patterns to compare against
        """
        
        fig, axes = plt.subplots(1,3,figsize=(12,5.3))
        fig.tight_layout(rect=[0.02, 0.09, 0.98, 0.96])
        axslider = plt.axes([0.15,0.06,0.75,0.03])
        
        
        def update_colorbar(im):
            # If the update brought the colorbar out of whack
            # (say, from clicking back in the navbar)
            # Holy fuck this was annoying. Sorry future for how
            # crappy this solution is.
            #if not np.allclose(im.colorbar.ax.get_xlim(),
            #                   (np.min(im.get_array()),
            #                    np.max(im.get_array()))):
            if hasattr(im, 'norecurse') and im.norecurse:
                im.norecurse=False
                return
            
            im.norecurse=True
            im.set_clim(vmin=np.min(im.get_array()),vmax=np.max(im.get_array()))

        def update(idx):
            idx = int(idx) % len(dataset)
            fig.pattern_idx = idx
            updating = True if len(axes[0].images) >= 1 else False
            
            inputs, output = dataset[idx]
            sim_data = self.forward(*inputs).detach().cpu().numpy()
            sim_data = sim_data
            meas_data = output.detach().cpu().numpy()
            if hasattr(self, 'mask') and self.mask is not None:
                mask = self.mask.detach().cpu().numpy()
            else:
                mask = 1
                
            if not updating:
                axes[0].set_title('Simulated')
                axes[1].set_title('Measured')
                axes[2].set_title('Difference')

                sim = axes[0].imshow(sim_data)
                meas = axes[1].imshow(meas_data * mask)
                diff = axes[2].imshow((sim_data-meas_data) * mask)

                cb1 = plt.colorbar(sim, ax=axes[0], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.1,fraction=0.1)
                cb1.ax.tick_params(labelrotation=20)
                cb1.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(sim))
                cb2 = plt.colorbar(meas, ax=axes[1], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.1,fraction=0.1)
                cb2.ax.tick_params(labelrotation=20)
                cb2.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(meas))
                cb3 = plt.colorbar(diff, ax=axes[2], orientation='horizontal',format='%.2e',ticks=ticker.LinearLocator(numticks=5),pad=0.1,fraction=0.1)
                cb3.ax.tick_params(labelrotation=20)
                cb3.ax.callbacks.connect('xlim_changed', lambda ax: update_colorbar(diff))
                
            else:

                sim = axes[0].images[-1]
                sim.set_data(sim_data)
                update_colorbar(sim)

                meas = axes[1].images[-1]
                meas.set_data(meas_data * mask)
                update_colorbar(meas)

                diff = axes[2].images[-1]
                diff.set_data((sim_data-meas_data) * mask)
                update_colorbar(diff)
                
                
        # This is dumb but the slider doesn't work unless a reference to it is
        # kept somewhere...
        self.slider = Slider(axslider, 'Pattern #', 0, len(dataset)-1, valstep=1, valfmt="%d")
        self.slider.on_changed(update)

        def on_action(event):
            if not hasattr(event, 'button'):
                event.button = None
            if not hasattr(event, 'key'):
                event.key = None
                
            if event.key == 'up' or event.button == 'up':
                update(fig.pattern_idx - 1)
            elif event.key == 'down' or event.button == 'down':
                update(fig.pattern_idx + 1)
            self.slider.set_val(fig.pattern_idx)
            plt.draw()

        fig.canvas.mpl_connect('key_press_event',on_action)
        fig.canvas.mpl_connect('scroll_event',on_action)
        update(0)
        


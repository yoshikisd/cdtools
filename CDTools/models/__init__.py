from __future__ import division, print_function, absolute_import

import torch as t
from torch.utils import data as torchdata
from matplotlib import pyplot as plt

#
# This is unrelated, but it will then be important to be able to save and load
# models easily from a predefined format. To be honest, this could just
# literally be by pickling the model. They could also be saved out as
# state_dicts or via torch.save. I think it's best to just save the whole
# model - I lose out on the modularity of just saving the state_dict, but
# I gain in it being easy to reload the non-learned aspects of the model,
# like the wavelength and sample geometry. Remember that it's important
# that the final outputs of the reconstructions are transferrable to other
# places
#


#
# For now, just save/load model via the built-in t.save() and t.load()
# functions
#



class CDIModel(t.nn.Module):
    """This base model defines all the functions that must be exposed for a valid CDIModel subclass

    Most of the functions only raise a NotImplementedError at this level and
    must be explicitly defined by any subclass. The functions required can be
    split into several subsections:

    Creation:
    from_dataset : a function to create a CDIModel from an appropriate CDataset

    Simulation:
    interaction : a function to simulate exit waves from experimental parameters
    forward_propagator : the propagator from the experiment plane to the detector plane
    backward_propagator : the propagator from the detector plane to the experiment plane
    measurement : a function to simulate the detector readout from a detector plane wavefront
    forward : predefined, the entire stacked forward model
    loss : the loss function to report and use for automatic differentiation
    simulation : predefined, simulates a stack of detector images from the forward model
    simulate_to_dataset : a function to create a CDataset from the simulation defined in the model

    Reconstruction:
    AD_optimize : predefined, a generic automatic differentiation reconstruction
    Adam_optimize : predefined,  sensible automatic differentiation reconstruction using ADAM

    The work of defining the various subclasses boils down to creating an
    appropriate implementation for this set of functions.
    """



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
        return self.measurement(self.forward_propagator(self.interaction(*args)))

    def loss(self, sim_data, real_data):
        raise NotImplementedError()


    # I know this is silly but it makes it clear this should be explicitly
    # overwritten
    def to(self, *args, **kwargs):
        super(CDIModel,self).to(*args,**kwargs)


    def simulate(self, args_list):
        return t.Tensor([self.forward(*args) for args in args_list])


    def simulate_to_dataset(self, args_list):
        raise NotImplementedError()


    def inspect(self):
        raise NotImplementedError()
    

    def AD_optimize(self, iterations, data_loader,  optimizer, scheduler=None):

        for it in range(iterations):
            loss = 0
            N = 0
            for inputs, patterns in data_loader:
                N += 1

                def closure():
                    optimizer.zero_grad()
                    sim_patterns = self.forward(*inputs)
                    if hasattr(self, 'mask'):
                        loss = self.loss(patterns,sim_patterns, mask=self.mask)
                    else:
                        loss = self.loss(patterns,sim_patterns)

                    loss.backward()
                    return loss

                loss += optimizer.step(closure).detach().cpu().numpy()

            loss /= N
            if scheduler is not None:
                scheduler.step(loss)

            yield loss


    def Adam_optimize(self, iterations, dataset, batch_size=15, lr=0.005, schedule=False):

        # Make a dataloader
        data_loader = torchdata.DataLoader(dataset, batch_size=batch_size,
                                           shuffle=True)

        # Define the optimizer
        optimizer = t.optim.Adam(self.parameters(), lr = lr)


        # Define the scheduler
        if schedule:
            scheduler = t.optim.ReduceLROnPlateau(optimizer, factor=0.2)
        else:
            scheduler = None

        return self.AD_optimize(iterations, data_loader, optimizer, scheduler=scheduler)


    def LBFGS_optimize(self, iterations, dataset, batch_size=None,
                       lr=0.1,history_size=2):

        # Make a dataloader
        if batch_size is not None:
            data_loader = torchdata.DataLoader(dataset, batch_size=batch_size,
                                               shuffle=True)
        else:
            data_loader = torchdata.DataLoader(dataset)


        # Define the optimizer
        optimizer = t.optim.LBFGS(self.parameters(),
                                  lr = lr, history_size=history_size)

        return self.AD_optimize(iterations, data_loader, optimizer)


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
        
        Args:
            dataset (torch.Dataset): Optional, a dataset matched to the model type
            update (bool) : Whether to update existing plots or plot new ones
        
        Returns:
            list : A list of figure numbers noting where the plots were plotted
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
            
        for idx, plots in enumerate(self.plot_list):
            if figs is None:
                fig = plt.figure()
                self.figs.append(fig)
            else:
                fig = figs[idx]
            
            name = plots[0]
            plotter = plots[1]
            # If a conditional is included in the plot
            try:
                if len(plots) >=3 and not plots[2](self):
                    continue
            except TypeError as e:
                if len(plots) >= 3 and not plots[2](self, dataset):
                    continue
            try:
                plotter(self,fig)
                plt.title(name)
            except TypeError as e:
                if dataset is not None:
                    try:
                        plotter(self, fig, dataset)
                        plt.title(name)
                    except (IndexError, KeyError, AttributeError) as e:
                        pass
            except (IndexError, KeyError, AttributeError) as e:
                pass
            if update:
                plt.draw()
                fig.canvas.start_event_loop(0.001)

        if first_update:
            plt.pause(0.05 * len(self.figs))





from CDTools.models.simple_ptycho import SimplePtycho
from CDTools.models.fancy_ptycho import FancyPtycho
from CDTools.models.incoherent_ptycho import IncoherentPtycho

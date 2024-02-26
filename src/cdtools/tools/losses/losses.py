"""Contains various loss functions to be used for optimization

It exposes three losses, one returning the mean squared amplitude error, one
that returns the mean squared intensity error, and one that returns the
maximum likelihood metric for a system with Poisson statistics.

"""

import torch as t

__all__ = ['amplitude_mse', 'intensity_mse', 'poisson_nll']


def amplitude_mse(intensities, sim_intensities, mask=None):
    """ Returns the mean squared error of a simulated dataset's amplitudes

    Calculates the mean squared error between a given set of 
    measured diffraction intensities and a simulated set.

    This function calculates the mean squared error between their
    associated amplitudes. Because this is not well defined for negative
    numbers, make sure that all the intensities are >0 before using this
    loss. Note that this is actually a sum-squared error, because this
    formulation makes it vastly simpler to compare error calculations
    between reconstructions with different minibatch size. I hope to
    find a better way to do this that is more honest with this
    cost function, though.

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    This is empirically the most useful loss function for most cases

    Parameters
    ----------
    intensities : torch.Tensor
        A tensor with measured detector values
    sim_intensities : torch.Tensor
        A tensor of simulated detector intensities
    mask : torch.Tensor
        A mask with ones for pixels to include and zeros for pixels to exclude

    Returns
    -------
    loss : torch.Tensor
        A single value for the mean amplitude mse
    """

    # I know it would be more efficient if this function took in the
    # amplitudes instead of the intensities, but I want to be consistent
    # with all the errors working off of the same inputs

    if mask is None:
        return t.sum((t.sqrt(sim_intensities) -
                      t.sqrt(intensities))**2)
    else:
        masked_intensities = intensities.masked_select(mask)
        return t.sum((t.sqrt(sim_intensities.masked_select(mask)) -
                      t.sqrt(masked_intensities))**2)


    
def intensity_mse(intensities, sim_intensities, mask=None):
    """ Returns the mean squared error of a simulated dataset's intensities

    Calculates the summed mean squared error between a given set of 
    diffraction intensities - the measured set of detector intensities -
    and a simulated set of diffraction intensities. This function
    calculates the mean squared error between the intensities.

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    Parameters
    ----------
    intensities : torch.Tensor
        A tensor with measured detector intensities.
    sim_intensities : torch.Tensor
        A tensor of simulated detector intensities
    mask : torch.Tensor
        A mask with ones for pixels to include and zeros for pixels to exclude

    Returns
    -------
    loss : torch.Tensor
        A single value for the mean intensity mse

    """
    if mask is None:
        return t.sum((sim_intensities - intensities)**2) \
            / intensities.view(-1).shape[0]
    else:
        masked_intensities = intensities.masked_select(mask)
        return t.sum((sim_intensities.masked_select(mask) -
                      masked_intensities)**2) \
                      / masked_intensities.shape[0]


    
def poisson_nll(
        intensities,
        sim_intensities,
        mask=None,
        eps=1e-6,
        subtract_min=False
    ):
    """ Returns the Poisson negative log likelihood for simulated intensities

    Calculates the overall Poisson maximum likelihood metric using
    diffraction intensities - the measured set of detector intensities -
    and a simulated set of intensities. This loss would be appropriate
    for detectors in a single-photon counting mode, with their output
    scaled to number of photons

    Note that this calculation ignores the log(intensities!) term in the
    full expression for Poisson negative log likelihood. This term doesn't
    change the calculated gradients so isn't worth taking the time to compute

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    The default value of eps is 1e-6 - a nonzero value here helps avoid
    divergence of the log function near zero.

    Parameters
    ----------
    intensities : torch.Tensor
        A tensor with measured detector intensities.
    sim_intensities : torch.Tensor
        A tensor of simulated detector intensities
    mask : torch.Tensor
        A mask with ones for pixels to include and zeros for pixels to exclude
    eps : float
        Optional, a small number to add to the simulated intensities
    
    Returns
    -------
    loss : torch.Tensor
        A single value for the poisson negative log likelihood

    """
    if mask is None:
        nll = t.sum(sim_intensities+eps -
                    t.xlogy(intensities,sim_intensities+eps)) \
                    / intensities.view(-1).shape[0]
        if subtract_min:
            nll -= t.sum(intensities - t.xlogy(intensities,intensities))
            
        return nll
    else:
        masked_intensities = intensities.masked_select(mask)
        masked_sims = sim_intensities.masked_select(mask)

        nll = t.sum(masked_sims + eps - \
                    t.xlogy(masked_intensities, masked_sims+eps)) \
                    / masked_intensities.shape[0]
        
        if subtract_min:
            nll -= t.nansum(masked_intensities - \
                    t.xlogy(masked_intensities, masked_intensities)) \
                    / masked_intensities.shape[0]

        return nll



def poisson_plus_fixed_nll(
        intensities,
        sim_intensities,
        fixed_nll,
        range,
        mask=None,
        eps=1e-6,
        subtract_min=False):
    """ Return a combined negative log likelihood for Poisson and fixed noise.

    This is unpublished as far as I know. First, it calculates the log
    likelihoods for all the possible true photon counts within (range)
    of the measured value, using the detector noise model.

    Next, it calculates the poisson log likelihood for each of those
    possible true photon counts within the minimum and maximum offset
    defined by range. It then compares them to the simulated poisson
    reciprocal arrival rate (simulated intensity). Finally, it combines
    them using a logsumexp to calculate the negative log likelihood for
    the combined noise model.
     
    This loss is appropriate for detectors with significant fixed readout
    noise which is independent of signal strength, after conversion from
    detector native units to units of photons. In the simple case,
    a gaussian nll can be passed to the fixed_nll function, but if the detector
    noise has been well characterized, 

    Note that this calculation ignores the log(intensities!) term in the
    full expression for Poisson negative log likelihood. This term doesn't
    change the calculated gradients so isn't worth taking the time to compute

    It can accept intensity and simulated intensity tensors of any shape
    as long as their shapes match, and the provided mask array can be
    broadcast correctly along them.

    The default value of eps is 1e-6 - a nonzero value here helps avoid
    divergence of the log function near zero.

    Parameters
    ----------
    intensities : torch.Tensor
        A tensor with measured detector intensities.
    sim_intensities : torch.Tensor
        A tensor of simulated detector intensities
    fixed_nll : function
        A function which calculates the fixed negative log likelihood part
    rang : tuple
        A pair (min, max) defining the relative search range for photon counts
    mask : torch.Tensor
        A mask with ones for pixels to include and zeros for pixels to exclude
    eps : float
        Optional, a small number to add to the simulated intensities
    
    Returns
    -------
    loss : torch.Tensor
        A single value for the poisson negative log likelihood

    """
    raise NotImplementedError()
    if mask is None:
        nll = t.sum(sim_intensities+eps -
                    t.xlogy(intensities,sim_intensities+eps)) \
                    / intensities.view(-1).shape[0]
        if subtract_min:
            nll -= t.sum(intensities - t.xlogy(intensities,intensities))
            
        return nll
    else:
        masked_intensities = intensities.masked_select(mask)
        masked_sims = sim_intensities.masked_select(mask)

        nll = t.sum(masked_sims + eps - \
                    t.xlogy(masked_intensities, masked_sims+eps)) \
                    / masked_intensities.shape[0]
        
        if subtract_min:
            nll -= t.nansum(masked_intensities - \
                    t.xlogy(masked_intensities, masked_intensities)) \
                    / masked_intensities.shape[0]

        return nll

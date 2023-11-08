import torch as t
import h5py
import pathlib
from cdtools.datasets import Ptycho2DDataset
from cdtools.tools import data as cdtdata

__all__ = ['PolarizationSweptPtycho2DDataset']


class PolarizationSweptPtycho2DDataset(Ptycho2DDataset):
    """The standard dataset for a 2D ptychography scan

    Subclasses datasets.CDataset

    This class loads and saves 2D ptychography scan data from .cxi files.
    It should save and load files compatible with most reconstruction
    programs, although it is only tested against SHARP.
    """
    def __init__(self, translations, patterns, polarization_indices,
                 polarization_states,
                 *args, **kwargs):
        """The __init__ function allows construction from python objects.

        The detector_geometry dictionary is defined to have the
        entries defined by the outputs of data.get_detector_geometry.

        Note that the created dataset object will not copy the data in the
        patterns parameter in order to avoid doubling the memory requiement
        for large datasets.

        Parameters
        ----------
        translations : array
            An nx3 array containing the probe translations at each scan point
        patterns : array
            An nxmxl array containing the full stack of measured diffraction
            patterns
        polarization_indices : array(int)
            A length-n array containing the index of the polarization state
            attached to each pattern
        polarization_states : array
            An mx2 array, where m is the number of polarization indices,
            encoding the polarzation state associated with each index
        axes : list(str)
            A list of names for the axes of the probe translations
        entry_info : dict
            A dictionary containing the entry_info metadata
        sample_info : dict
            A dictionary containing the sample_info metadata
        wavelength : float
            The wavelength of light used in the experiment
        detector_geometry : dict
            A dictionary containing the various detector geometry
            parameters
        mask : array
            A mask for the detector, defined as 1 for live pixels, 0
            for dead
        background : array
            An initial guess for the not-previously-subtracted
            detector background
        intensities : array
            A list of measured shot-to-shot intensities
        """
        
        super().__init__(translations, patterns,
                                                      *args, **kwargs)
        
        self.polarization_indices = t.tensor(polarization_indices, dtype=t.int32)
        self.polarization_states = t.tensor(polarization_states, dtype=t.float32)

        
    def _load(self, index):
        """ Internal function to load data

        This function is used internally by the global __getitem__ function
        defined in the base class, which handles moving data around when
        the dataset is (for example) storing the data on the CPU but
        getting data as GPU tensors.

        It loads data in the format (inputs, output)

        The inputs for a 2D ptychogaphy data set are:

        1) The indices of the patterns to use
        2) The recorded probe positions associated with those points
        3) The angles of the polarizers if polarized=True

        Parameters
        ----------
        index (polarized=False): int or slice
             The index or indices of the scan points to use

        index (polarized=True): 
            tuple ((phi1, phi2), ind) 

            ind - index or indices of the scan points to use (in a (phi1, phi2) polarization state), int or slice
            (phi1, phi2) - angles of the 1st and 2nd polarizers, ints

        Returns
        -------
        inputs : tuple
             A tuple of the inputs to the related forward models
             if polarized: inputs = ((phi1, phi2), ind, transl[(phi1, phi2)][ind])
        outputs : tuple
             The output pattern or stack of output patterns
        """
        return ((index, self.translations[index],
                 self.polarization_indices[index]),
                self.patterns[index])

    def to(self, *args, **kwargs):
        """Sends the relevant data to the given device and dtype

        This function sends the stored translations, patterns,
        mask and background to the specified device and dtype

        Accepts the same parameters as torch.Tensor.to
        """
        super().to(*args, **kwargs)
        self.polarization_states = self.polarization_states.to(*args, **kwargs)


    # It sucks that I can't reuse the base factory method here,
    # perhaps there is a way but I couldn't figure it out.
    @classmethod
    def from_cxi(cls, cxi_file, cut_zeros=True):
        """Generates a new PolarizationSweptPtycho2DDataset from a .cxi file 

        This generates a new PolarizationSweptPtycho2DDataset from a .cxi file
        storing the 2D ptychography scan.

        Parameters
        ----------
        file : str, pathlib.Path, or h5py.File
            The .cxi file to load from
        cut_zeros : bool
            Default True, whether to set all negative data to zero

        Returns
        -------
        dataset : PolarizationSweptPtycho2DDataset
            The constructed dataset object
        """

        # If a bare string is passed
        if isinstance(cxi_file, str) or isinstance(cxi_file, pathlib.Path):
            with h5py.File(cxi_file, 'r') as f:
                return cls.from_cxi(f, cut_zeros=cut_zeros)

        # Generate a base dataset
        dataset = Ptycho2DDataset.from_cxi(cxi_file, cut_zeros=cut_zeros)

        # Mutate the class to this subclass (PolarizedPtycho2DDataset)
        dataset.__class__ = cls

        # Now, we save out the polarizer and analyzer states
        polarization_indices = cdtdata.get_shot_to_shot_info(cxi_file, 'polarization_indices')
        polarization_states = cdtdata.get_entry_info(cxi_file, 'polarization_states')

        dataset.polarization_indices = t.tensor(polarization_indices, dtype=t.int32)
        dataset.polarization_states = t.tensor(polarization_states, dtype=t.complex64)
        
        return dataset

    def to_cxi(self, cxi_file, polarized=False):
        """Saves out a PolarizationSweptPtycho2DDataset as a .cxi file

        This function saves all the compatible information in a
        PolarizationSweptPtycho2DDataset object into a .cxi file. This saved
        .cxi file should be compatible with any standard .cxi file based
        reconstruction tool, such as SHARP.

        Parameters
        ----------
        cxi_file : str, pathlib.Path, or h5py.File
            The .cxi file to write to
        """

        # If a bare string is passed
        if isinstance(cxi_file, str) or isinstance(cxi_file, pathlib.Path):
            with cdtdata.create_cxi(cxi_file) as f:
                return self.to_cxi(f)

        # This saves the translations, patterns, etc.
        super().to_cxi(cxi_file)

        # Now, we save out the polarization states
        cdtdata.add_shot_to_shot_info(cxi_file, self.polarization_indices,
                                      'polarization_indices')
        cdtdata.add_entry_info(cxi_file, self.polarization_states,
                                      'polarization_states')

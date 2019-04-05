from __future__ import division, print_function, absolute_import

import h5py
import numpy as np
import numbers
import datetime
import dateutil.parser
import torch as t
from contextlib import contextmanager

__all__ = ['get_entry_info',
           'get_sample_info',
           'get_wavelength',
           'get_detector_geometry',
           'get_mask',
           'get_data',
           'get_ptycho_translations',
           'create_cxi',
           'add_entry_info',
           'add_sample_info',
           'add_source',
           'add_detector',
           'add_mask',
           'add_data',
           'add_ptycho_translations']

#
#
# I will put here some thoughts about how to load data into this program.
#
# 
# The reconstructions should have the ability to generate datasets.
# So you could write a reconstruction engine and then it would be
# able to simulate data directly in the engine for you to use as a
# reconstruction
#
# I don't even think there needs to be a loading tool for loading cxi files
# because there isn't really a better method beyond just loading the
# file into an h5py object. This file could host the simple cxi file
# browser, perhaps. But I think the reality is that we need individual
# loaders for each kind of experiment. Perhaps we could put some basic
# reuseable tools for inspecting cxi-type h5 files in this file. 
#
#
# Then, there can be some more sophisticated tools that load data for
# specific use cases that are common - loading data for a 2D CDI experiment,
# loading data for a 2D Ptycho experiment, loading data for Bragg Ptycho in
# 3D, loading data for a 3D CDI experiment, etc.
#
#
# Perhaps one good way to package this is for the kind of data associated
# with any particular experiment to have it's own kind of dataset or view.
# So there would be a "2D Ptychography" data viewer, which would contain
# all the measured data that comes from a 2D ptychography experiment.
# The specialized functions would plop out these data viewers, and the
# reconstruction classes could be designed around a particular kind of
# viewer with the most general kind just requiring a generic data viewer.
#
# Data viewers could have simple tools like the ability to send themselves
# to the GPU, CPU, change the datatype, etc. I think the most generic thing
# is as a subclass of the torch Data objects, where they would for each slice
# return the index, a set of defining parameters (translation, angle, energy,
# whatever), and a diffraction pattern. They would also have a "setup"
# attribute, or "metadata", or whatever you'd want to call it, that contain
# the various fixed experimental parameters (energy, distance, etc.)
# 
# And I think the cxi visualizer should really go into it's own script,
# because it's not a reuseable component.
#



#
# Functions to inspect the basic attributes of a cxi file represented as an
# h5 file object
#


def get_entry_info(cxi_file):
    """Returns a dictionary with the basic metadata from the cxi file's entry_1 attribute

    String type metadata is read out as a string, and datetime metadata
    is converted to python datetime objects if the string is properly
    formatted.
    
    Args:
        cxi_file (h5py.File) : a file object to be read

    Returns:
        dict : A dictionary with basic metadata defined in the cxi file

    """
    e1 = cxi_file['entry_1']
    metadata_attrs = ['title',
                      'experiment_identifier',
                      'experiment_description',
                      'program_name']
    metadata = {attr: str(e1[attr][()].decode()) for attr in metadata_attrs
                if attr in e1}
    datetime_attrs = ['start_time',
                      'end_time']
    for attr in datetime_attrs:
        if attr in e1:
            try:
                metadata[attr] = dateutil.parser.parse(str(e1[attr][()].decode()))
            except ValueError:
                metadata[attr] = str(e1[attr][()].decode())
    return metadata


def get_sample_info(cxi_file):
    """Returns a dictionary with the basic metadata from the cxi file's entry_1/sample_1 attribute
    
    Args:
        cxi_file (h5py.File) : a file object to be read

    Returns:
        dict : A dictionary with basic metadata from the sample defined in the cxi file

    """
    if 'entry_1/sample_1' not in cxi_file:
        return None
    
    s1 = cxi_file['entry_1/sample_1']
    metadata_attrs = ['name','description','unit_cell_group']

    metadata = {}
    for attr in metadata_attrs:
        # Somehow different ways of saving can lead to different ways to
        # decode it here, so we try both
        if attr in s1:
            try:
                metadata[attr] = str(s1[attr][()].decode())
            except AttributeError as e:
                metadata[attr] = str(np.array(s1[attr][:])[0].decode())
    
    float_attrs = ['concentration',
                   'mass',
                   'temperature',
                   'thickness',
                   'unit_cell_volume']
    for attr in float_attrs:
        if attr in s1:
            metadata[attr] = np.float32(s1[attr][()])

    if 'unit_cell' in s1:
        metadata['unit_cell'] = np.array(s1['unit_cell']).astype(np.float32)

    # TODO: Add my nonstandard "surface normal" attribute here
    
    # TODO: I should add the sample geometry as a valid metadata that can
    # be copied over
    
    # Check if the metadata is empty
    if metadata == {}:
        metadata = None
    
    return metadata


def get_wavelength(cxi_file):
    """Returns the wavelength of the source defined in the cxi file object, in m
    
    Args:
        cxi_file (h5py.File) : a file object to be read
    
    Returns:
        np.float32 : The wavelength of the source defined in the cxi file
    """
    i1 = cxi_file['entry_1/instrument_1']
    if 'source_1/wavelength' in i1:
        wavelength = np.float32(i1['source_1/wavelength'])
    elif 'source_1/energy' in i1:
        energy = np.float32(i1['source_1/energy'])
        wavelength = 1.9864459e-25 / energy
    else:
        raise KeyError('Neither Wavelength or Energy Defined in provided .cxi File')
    
    return wavelength


def get_detector_geometry(cxi_file):
    """Returns a standardized description of the detector geometry defined in the cxi file object

    It makes intelligent assumptions based on the definitions in the cxi
    file definition. The standardized description of the geometry that it
    outputs includes the sample to detector distance, the corner location
    of the detector, and the basis vectors defining the detector. It can
    only handle detectors defined as rectangular grids of pixels.
    
    The distance and corner_location values are technically overdetermining
    the detector location, but for many experiments (particularly
    transmission experiments), the distance is needed and the exact 
    corner location is not. If the corner location is not reported in
    the cxi file, no attempt will be made to calculate it.

    Args:
        cxi_file (h5py.File) : a file object to be read
    
    Returns:
        distance (np.float32) : The sample to detector distance, in m
        basis_vectors (np.array) : The basis vectors for the detector
        corner_location (np.array) : The location of the (0,0) pixel in the detector

    """
    i1 = cxi_file['entry_1/instrument_1']
    d1 = i1['detector_1']

    if 'detector_1/basis_vectors' in i1:
        basis_vectors = np.array(d1['basis_vectors'])
    else:
        # This whole thing just to account for all the ways people can
        # implicitly define the x or y pixel size for a detector. I've
        # seen too many of these in the wild, unfortunately...
        try:
            x_pixel_size = np.float32(d1['x_pixel_size'])
        except:
            x_pixel_size = None
        try:
            y_pixel_size = np.float32(d1['y_pixel_size'])
        except:
            y_pixel_size = None

        if x_pixel_size is None and y_pixel_size is not None:
            x_pixel_size = y_pixel_size
        elif x_pixel_size is not None and y_pixel_size is None:
            y_pixel_size = x_pixel_size
        if x_pixel_size is None and y_pixel_size is None:
            raise KeyError('Detector pixel size not defined in file.')
        basis_vectors = np.array([[0,-y_pixel_size,0],
                                  [-x_pixel_size,0,0]]).transpose()
        
    try:
        distance = np.float32(d1['distance'])
    except:
        distance = None
    try:
        corner_position = np.array(d1['corner_position'])
    except:
        corner_position = None
    
    # Don't pretend to calculate corner position from distance if it's
    # if it's not defined, but do calculate distance from corner position
    # if distance is not defined. If neither is defined, then raise
    # an error.
    if distance is None and corner_position is not None:
        detector_normal = np.cross(basis_vectors[:,0],
                                   basis_vectors[:,1])
        detector_normal /= np.linalg.norm(detector_normal)
        distance = np.linalg.norm(np.dot(corner_position, detector_normal))
        
    if distance is None and corner_position is not None:
        raise KeyError('Neither sample to detector distance or corner position is defined in file.')

    return distance, basis_vectors, corner_position


def get_mask(cxi_file):
    """Returns the detector mask defined in the cxi file object

    This function converts from the format specified in the cxi file
    definition to a simple on/off mask, where a value of 1 defines a
    good pixel (on) and a value of 0 defines a bad pixel (off).

    If any bit is set in the mask at all, it will be defined as a bad
    pixel, with the exception of pixels marked exactly as 0x00001000,
    which is defined to mean that the pixel has signal above the
    background. These pixels are treated as on pixels
    
    Args:
        cxi_file (h5py.File) : a file object to be read
    
    Returns:
        np.array : An array storing the mask from the cxi file
    """
    
    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1/mask' in i1:
        mask = np.array(i1['detector_1/mask']).astype(np.uint32)
        mask_on = np.equal(mask,np.uint32(0))
        mask_has_signal = np.equal(mask,np.uint32(0x00001000))
        return np.logical_or(mask_on,mask_has_signal).astype(np.uint8)
    else:
        return None


def get_data(cxi_file, cut_zeroes = True):
    """Returns an array with the full stack of detector data defined in the cxi file object

    This function will make sure to check all the various places that it's
    okay to store the data in, to ensure that it can find the data regardless
    of whether the creator of the .cxi file has remembered to link the data
    to all the required locations.

    It will return the data array in whatever shape it's defined in.
    
    It will also read out the axes attribute of the data into a list
    of strings
    
    Args:
        cxi_file (h5py.File) : a file object to be read
    
    Returns:
        np.array : An array storing the data defined in the cxi file
        list : A list of the axes defined in the axes attribute, if any
    """
    # Possible locations for the data
    # 
    # entry_1/detector_1/data
    if 'entry_1/data_1/data' in cxi_file:
        pull_from = 'entry_1/data_1/data'
    elif 'entry_1/instrument_1/detector_1/data' in cxi_file:
        pull_from = 'entry_1/instrument_1/detector_1/data'
    else:
        raise KeyError('Data is not defined within cxi file')
    data = np.array(cxi_file[pull_from]).astype(np.float32)

    if cut_zeroes:
        data[data < 0] = 0
        
    if 'axes' in cxi_file[pull_from].attrs:
        axes = str(cxi_file[pull_from].attrs['axes'].decode()).split(':')
        axes = [axis.strip().lower() for axis in axes]
    else:
        axes = None

    return data, axes



def get_ptycho_translations(cxi_file):
    """Gets an array of x,y,z translations, if such an array has been defined in the file

    It negates the translations, because the CXI file format is designed
    to specify translations of the samples and the CDTools code specifies
    translations of the optics.
    
    Args:
        cxi_file (h5py.File) : a file object to be read
    
    Returns:
        np.array : An array storing the translations defined in the cxi file
        list : A list of the axes defined in the axes attribute, if any
    """
    
    if 'entry_1/data_1/translation' in cxi_file:
        pull_from = 'entry_1/data_1/translation'
    elif 'entry_1/sample_1/geometry_1/translation' in cxi_file:
        pull_from = 'entry_1/sample_1/geometry_1/translation'
    elif 'entry_1/instrument_1/detector_1/translation' in cxi_file:
        pull_from = 'entry_1/instrument_1/detector_1/translation'
    else:
        raise KeyError('Translations are not defined within cxi file')

    translations = -np.array(cxi_file[pull_from]).astype(np.float32)
    return translations
    


#
# It might be useful to make some helper functions to help write cxi files
#


def create_cxi(filename):
    """Creates a new cxi file with a single entry group

    Args:
        filename (str) : The path at which to create the file    
    """
    file_obj = h5py.File(filename,'w')
    file_obj.create_dataset('cxi_version', data=160)  
    file_obj.create_dataset('number_of_entries',data=1)
    e1f = file_obj.create_group('entry_1')
    return file_obj


def add_entry_info(cxi_file, metadata):
    """Adds a dictionary of entry metadata to the entry_1 group of a cxi file object
    
    Args:
        cxi_file (h5py.File) : The file to add the info to
        metadata (dict) : A dictionary containing all the metadata to be stored
    """
    # Just the string and datetime types should be relevant but all are
    # included in case the cxi spec becomes more permissive
    for key, value in metadata.items():
        if isinstance(value,(str,bytes)):
            cxi_file['entry_1'][key] = np.string_(value)
        elif isinstance(value, datetime.datetime):
            cxi_file['entry_1'][key] = np.string_(value.isoformat())
        elif isinstance(value, numbers.Number):
            si[key] = value
        elif isinstance(value, (np.ndarray,list,tuple)):
            s1.create_dataset(key, data=np.asarray(value))
        elif isinstance(value, t.Tensor):
            asnumpy = value.detach().cpu().numpy()
            cxi_file['entry_1'].create_dataset(key, data=asnumpy)


def add_sample_info(cxi_file, metadata):
    """Adds a dictionary of entry metadata to the entry_1/sample_1 group of a cxi file object

    This function will create the sample_1 attribute if it doesn't already exist
    
    Args:
        cxi_file (h5py.File) : The file to add the info to
        metadata (dict) : A dictionary containing all the metadata to be stored
    """
    if 'entry_1/sample_1' not in cxi_file:
        cxi_file['entry_1'].create_group('sample_1')
    s1 = cxi_file['entry_1/sample_1']
        
    for key, value in metadata.items():
        if isinstance(value,(str,bytes)):
            s1[key] = np.string_(value)
        elif isinstance(value, datetime.datetime):
            s1[key] = np.string_(value.isoformat())
        elif isinstance(value, numbers.Number):
            s1[key] = value
        elif isinstance(value, (np.ndarray,list,tuple)):
            s1.create_dataset(key, data=np.asarray(value))
        elif isinstance(value, t.Tensor):
            asnumpy = value.detach().cpu().numpy()
            s1.create_dataset(key, data=asnumpy)
    

def add_source(cxi_file, wavelength):
    """Adds the entry_1/source_1 group to a cxi file object
    
    It stores the energy and wavelength attributes in the source_1 group,
    given a wavelength to define them from.

    Args:
        cxi_file (h5py.File) : The file to add the source to
        wavelength (float) : The wavelength of light
    """
    if 'entry_1/instrument_1' not in cxi_file:
        cxi_file['entry_1'].create_group('instrument_1')
    i1 = cxi_file['entry_1/instrument_1']
    if 'source_1' not in i1:
        i1.create_group('source_1')
    s1 = i1['source_1']
    s1['wavelength'] = np.float32(wavelength)
    s1['energy'] = np.float32(1.9864459e-25 / wavelength)



def add_detector(cxi_file, distance, basis, corner=None):
    """Adds the entry_1/instrument_1/detector_1 group to a cxi file object
    
    It will define all the relevant parameters - distance, pixel size,
    detector basis, and corner position (if relevant) based on the provided
    information

    Args:
        cxi_file (h5py.File) : The file to add the detector to
        distance (float) : The sample to detector distance
        basis (array_like) : The detector basis
        corner (array_like) : Optional, the corner position of the detector
    
    """
    if 'entry_1/instrument_1' not in cxi_file:
        cxi_file['entry_1'].create_group('instrument_1')
    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1' not in i1:
        i1.create_group('detector_1')
    d1 = i1['detector_1']

    d1['distance'] = np.float32(distance)
    d1['x_pixel_size'] = np.linalg.norm(basis[:,1])
    d1['y_pixel_size'] = np.linalg.norm(basis[:,0])
    if isinstance(basis, t.Tensor):
        basis = basis.detach().cpu().numpy()
    d1.create_dataset('basis_vectors', data=basis)
    if corner is not None:
        if isinstance(corner, t.Tensor):
            corner = corner.detach().cpu().numpy()
        d1.create_dataset('corner_position',data=corner)


def add_mask(cxi_file, mask):
    """Adds the specified mask to the cxi file

    It places the mask into the mask dataset under 
    entry_1/instrument_1/detector_1. The internal mask is defined 
    simply as a 1 for an "on" pixel and a 0 for an "off" pixel, and
    the saved mask is exactly the opposite. This is simpler than the
    most general mask allowed by the cxi file format but it captures the
    distinction between pixels to be used and pixels not to be used.
    
    Args:
        cxi_file (h5py.File) : The file to add the mask to    
        mask (array_like) : The mask to save out to the file
    """
    
    if 'entry_1/instrument_1' not in cxi_file:
        cxi_file['entry_1'].create_group('instrument_1')
    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1' not in i1:
        i1.create_group('detector_1')
    d1 = i1['detector_1']
    if isinstance(mask, t.Tensor):
        mask = mask.detach().cpu().numpy()
    
    mask_to_save = np.zeros(mask.shape).astype(np.uint32)
    mask_to_save[mask == 0] = 1
    d1.create_dataset('mask',data=mask_to_save)


def add_data(cxi_file, data, axes=None):
    """Adds the specified data to the cxi file
    
    It will add the data unchanged to the file, placing it in two spots:
    
    1) The entry_1/instrument_1/detector_1/data path
    2) A softlink at entry_1/data_1/data

    Args:
        cxi_file (h5py.File) : The file to add the data to
        data (array_like) : The data to be saved
        axes (list) : Optional, a list of axis names to be saved in the axes attribute
    """
    if 'entry_1/data_1' not in cxi_file:
        cxi_file['entry_1'].create_group('data_1')
    data1 = cxi_file['entry_1/data_1']
    
    if 'entry_1/instrument_1' not in cxi_file:
        cxi_file['entry_1'].create_group('instrument_1')
    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1' not in i1:
        i1.create_group('detector_1')
    det1 = i1['detector_1']

    if isinstance(data, t.Tensor):
        data = data.detach().cpu().numpy()

    det1.create_dataset('data', data=data)
    data1['data'] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')

    if axes is not None:
        if isinstance(axes, list):
            axes_str = ':'.join(axes)
        else:
            axes_str = str(axes)
        det1['data'].attrs['axes'] = np.string_(axes_str)


def add_ptycho_translations(cxi_file, translations):
    """Adds the specified translations to the cxi file
    
    It will add the translations to the file, negating them to conform to
    the standard in cxi files that the translations refer to the object's
    translation.
    
    It will generally store them in 3 places:

    1) The entry_1/sample_1/geometry_1/translation path
    2) A softlink at entry_1/data_1/translation
    3) A softlink at entry_1/instrument_1/detector_1/translation
    
    Args:
        cxi_file (h5py.File) : The file to add the translations to
        translations (array_like) : The translations to be saved
    """

    if 'entry_1/sample_1' not in cxi_file:
        cxi_file['entry_1'].create_group('sample_1')
    s1 = cxi_file['entry_1/sample_1']

    if 'geometry_1' not in s1:
        s1.create_group('geometry_1')
    g1 = s1['geometry_1']
    
    if 'entry_1/data_1' not in cxi_file:
        cxi_file['entry_1'].create_group('data_1')
    data1 = cxi_file['entry_1/data_1']
    
    if 'entry_1/instrument_1' not in cxi_file:
        cxi_file['entry_1'].create_group('instrument_1')
    i1 = cxi_file['entry_1/instrument_1']
    if 'detector_1' not in i1:
        i1.create_group('detector_1')
    det1 = i1['detector_1']


    if isinstance(translations, t.Tensor):
        translations = translations.detach().cpu().numpy()

    # accounting for the different definition between cxi files and
    # CDTools
    translations = -translations
    
    g1.create_dataset('translation', data=translations)
    data1['translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
    det1['translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')

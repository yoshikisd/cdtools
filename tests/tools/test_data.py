from __future__ import division, print_function, absolute_import

from CDTools.tools import data
import numpy as np
import torch as t
import h5py
import pytest
import os
import datetime
import numbers
from pathlib import Path


#
# First, we write a few fixtures to generate specific data files we
# want with deliberate pathologies.
#
# * Energy defined but no wavelength
# * Wavelength defined but no energy
# * Distance and pixel pitch defined but no corner position or basis
# * Corner position and basis defined but no distance or pixel pitch
# * No mask defined
# * Mask defined
#
# Each file will get a fixture that loads the cxi file but also loads
# a dictionary with the relevant information for the cxi file
#
#


# This just grabs the directory whose name matches the file we're running
@pytest.fixture(scope='module')
def datadir(request):
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)
    return Path(test_dir)


@pytest.fixture(scope='module')
def ptycho_cxi_1():
    """Creates an example file for CXI ptychography. This file is defined
    to have everything done as correctly as possible with lots of attributes
    defined. It will return both a dictionary describing what is expected
    to be loaded and a file with the data stored in it.
    """
    
    expected = {}
    f = h5py.File('ptycho_cxi_1',driver='core',backing_store=False)

    # Start by defining the basic structure
    f.create_dataset('cxi_version', data=150)  
    f.create_dataset('number_of_entries',data=1)

    # Then define a bunch of metadata for entry_1
    e1f = f.create_group('entry_1')
    expected['entry metadata'] = {}
    e1e = expected['entry metadata']
    e1e['start_time'] = datetime.datetime.now()
    e1f['start_time'] = np.string_(e1e['start_time'].isoformat())
    e1e['end_time'] = datetime.datetime.now()
    e1f['end_time'] = np.string_(e1e['end_time'].isoformat())
    e1e['experiment_identifier'] = 'Fake Experiment 1'
    e1f['experiment_identifier'] = np.string_(e1e['experiment_identifier'])
    e1e['experiment_description'] = 'A fully defined ptychography experiment to test the data loading'
    e1f['experiment_description'] = np.string_(e1e['experiment_description'])
    e1e['program_name'] = 'CDTools'
    e1f['program_name'] = np.string_(e1e['program_name'])
    e1e['title'] = 'The one experiment we did'
    e1f['title'] = np.string_(e1e['title'])

    # Set up the sample info
    s1f = e1f.create_group('sample_1')
    expected['sample info'] = {}
    s1e = expected['sample info']
    s1e['name'] = 'Fake Sample'
    s1f['name'] = np.string_(s1e['name'])
    s1e['description'] = 'A sample that isn\'t real'
    s1f['description'] = np.string_(s1e['description'])
    s1e['unit_cell_group'] = 'P1'
    s1f['unit_cell_group'] = np.string_(s1e['unit_cell_group'])
    s1e['concentration'] = np.float32(np.random.rand())
    s1f['concentration'] = s1e['concentration']
    s1e['mass'] = np.float32(np.random.rand())
    s1f['mass'] = s1e['mass']
    s1e['temperature'] = np.float32(np.random.rand()*100)
    s1f['temperature'] = s1e['temperature']
    s1e['thickness'] = np.float32(np.random.rand()*1e-7)
    s1f['thickness'] = s1e['thickness']
    s1e['unit_cell_volume'] = np.float32(np.random.rand() * 1e-27)
    s1f['unit_cell_volume'] = s1e['unit_cell_volume']
    s1e['unit_cell'] = np.array([1,1,1,90,90,90]).astype(np.float32)
    s1f.create_dataset('unit_cell',data = s1e['unit_cell'])

    i1f = e1f.create_group('instrument_1')
    source1f = i1f.create_group('source_1')

    energy = np.float32(1.3618e-16) #Joules, = 850 eV
    source1f['energy'] = energy
    expected['wavelength'] = np.float32(1.9864459e-25) / energy
    source1f['wavelength'] = expected['wavelength'] 

    d1f = i1f.create_group('detector_1')
    expected['detector'] = {}
    d1e = expected['detector']
    d1e['distance'] = np.float32(0.3)
    d1f['distance'] = d1e['distance']
    d1e['basis'] = np.array([[0,-30e-6,0],
                             [-20e-6,0,0]]).astype(np.float32).transpose()
    d1f.create_dataset('basis_vectors',data=d1e['basis'])
    d1f['x_pixel_size'] = np.float32(20e-6)
    d1f['y_pixel_size'] = np.float32(30e-6)
    d1e['corner'] = np.array((2550e-6,3825e-6,0.3)).astype(np.float32)
    d1f.create_dataset('corner_position', data=d1e['corner'])

    # Remember the format for the CXI file differs from the format used
    # internally
    mask = np.zeros((100,256,256)).astype(np.uint32)
    expected['mask'] = np.ones((100,256,256)).astype(np.uint8)
    d1f.create_dataset('mask',data=mask)

    data1f = e1f.create_group('data_1')
    
    data = np.random.rand(100,256,256).astype(np.float32)
    expected['data'] = data
    d1f.create_dataset('data',data=data)
    data1f['data'] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')

    d1f['data'].attrs['axes'] = np.string_('translation:y:x')
    expected['axes'] = ['translation','y','x']

    g1f = s1f.create_group('geometry_1')
    translations = np.arange(300).reshape((100,3)).astype(np.float32)
    g1f.create_dataset('translation',data=translations)
    data1f['translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
    d1f['translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
    expected['translations'] = -translations.transpose()
    
    yield f, expected

    f.close()


@pytest.fixture(scope='module')
def ptycho_cxi_2():
    """Creates an example file for CXI ptychography. This file is defined
    to have a subset of things missing. In particular, it:

    * Defines the wavelength but not the energy
    * Defines the corner position but not the sample-detector distance
    * Defines pixel sizes but no basis vectors
    * Doesn't define a mask
    * Only defines data in the relevant places, not under data_1
    * Doesn't explicitly define axes for the data arrays
    * Is missing many allowed metadata attributes

    """
    
    expected = {}
    f = h5py.File('ptycho_cxi_2',driver='core',backing_store=False)

    # Start by defining the basic structure
    f.create_dataset('cxi_version', data=150)  
    f.create_dataset('number_of_entries',data=1)

    # Then define a bunch of metadata for entry_1
    e1f = f.create_group('entry_1')
    expected['entry metadata'] = {}
    e1e = expected['entry metadata']
    e1e['title'] = 'The one experiment we did'
    e1f['title'] = np.string_(e1e['title'])

    # Set up the sample info
    s1f = e1f.create_group('sample_1')
    expected['sample info'] = {}
    s1e = expected['sample info']
    s1e['temperature'] = np.float32(np.random.rand()*100)
    s1f['temperature'] = s1e['temperature']

    i1f = e1f.create_group('instrument_1')
    source1f = i1f.create_group('source_1')

    energy = np.float32(1.3618e-16) #Joules, = 850 eV
    expected['wavelength'] = np.float32(1.9864459e-25) / energy
    source1f['wavelength'] = expected['wavelength'] 

    d1f = i1f.create_group('detector_1')
    expected['detector'] = {}
    d1e = expected['detector']
    d1e['distance'] = np.float32(0.3)
    d1e['basis'] = np.array([[0,-30e-6,0],
                             [-20e-6,0,0]]).astype(np.float32).transpose()
    d1f['x_pixel_size'] = np.float32(20e-6)
    d1f['y_pixel_size'] = np.float32(30e-6)
    d1e['corner'] = np.array((2550e-6,3825e-6,0.3)).astype(np.float32)
    d1f.create_dataset('corner_position', data=d1e['corner'])

    # Remember the format for the CXI file differs from the format used
    # internally
    expected['mask'] = None

    data1f = e1f.create_group('data_1')
    
    data = np.random.rand(100,256,256).astype(np.float32)
    expected['data'] = data
    d1f.create_dataset('data',data=data)

    expected['axes'] = None

    g1f = s1f.create_group('geometry_1')
    translations = np.arange(300).reshape((100,3)).astype(np.float32)
    g1f.create_dataset('translation',data=translations)
    expected['translations'] = -translations.transpose()
    
    yield f, expected

    f.close()


@pytest.fixture(scope='module')
def ptycho_cxi_3():
    """Creates an example file for CXI ptychography. This file is defined
    to have a different subset of information missing. In particular, it:

    * Has no sample_1 group
    * Defines the energy but not the wavelength of light
    * Has the data only defined under the data_1 group, not in the relevant places
    * Defines the detector basis but no pixel sizes
    * Defines a mask as all pixels flagged as "above the background"
    * Defines the sample to detector distance but no corner location
    * Is missing some of the allowed metadata
    """
    
    expected = {}
    f = h5py.File('ptycho_cxi_3',driver='core',backing_store=False)

    # Start by defining the basic structure
    f.create_dataset('cxi_version', data=150)  
    f.create_dataset('number_of_entries',data=1)

    # Then define a bunch of metadata for entry_1
    e1f = f.create_group('entry_1')
    expected['entry metadata'] = {}
    e1e = expected['entry metadata']
    e1e['start_time'] = datetime.datetime.now()
    e1f['start_time'] = np.string_(e1e['start_time'].isoformat())
    e1e['end_time'] = datetime.datetime.now()
    e1f['end_time'] = np.string_(e1e['end_time'].isoformat())

    # Set up the sample info
    expected['sample info'] = None

    i1f = e1f.create_group('instrument_1')
    source1f = i1f.create_group('source_1')

    energy = np.float32(1.3618e-16) #Joules, = 850 eV
    source1f['energy'] = energy
    expected['wavelength'] = np.float32(1.9864459e-25) / energy

    d1f = i1f.create_group('detector_1')
    expected['detector'] = {}
    d1e = expected['detector']
    d1e['distance'] = np.float32(0.3)
    d1f['distance'] = d1e['distance']
    d1e['basis'] = np.array([[0,-30e-6,0],
                             [-20e-6,0,0]]).astype(np.float32).transpose()
    d1f.create_dataset('basis_vectors',data=d1e['basis'])
    d1e['corner'] = None

    # Remember the format for the CXI file differs from the format used
    # internally
    mask = np.ones((100,256,256)).astype(np.uint32) * 0x00001000
    expected['mask'] = np.ones((100,256,256)).astype(np.uint8)
    d1f.create_dataset('mask',data=mask)

    data1f = e1f.create_group('data_1')
    
    data = np.random.rand(100,256,256).astype(np.float32)
    expected['data'] = data
    data1f.create_dataset('data',data=data)

    data1f['data'].attrs['axes'] = np.string_('translation:y:x')
    expected['axes'] = ['translation','y','x']

    translations = np.arange(300).reshape((100,3)).astype(np.float32)
    data1f.create_dataset('translation',data=translations)
    expected['translations'] = -translations.transpose()
    
    yield f, expected

    f.close()


# As specific issues start to crop up with loading CXI files from different
# beamlines, put a fixture here that replicates the issue so that we can
# ensure compatibility with many beamlines
#


@pytest.fixture(scope='module')
def test_ptycho_cxis(ptycho_cxi_1, ptycho_cxi_2, ptycho_cxi_3):
    """Loads a list of tuples of ptychography CXI files and dictionaries,
    describing the expected output from various functions on being called
    on the cxi files.
    """
    return [ptycho_cxi_1, ptycho_cxi_2, ptycho_cxi_3]
    


#
# Now we have a bunch of tests of the data loading capabilities
#



def test_get_entry_info(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        entry_info = data.get_entry_info(cxi)
        for key in expected['entry metadata']:
            assert entry_info[key] == expected['entry metadata'][key]

        
def test_get_sample_info(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        sample_info = data.get_sample_info(cxi)
        if sample_info is None and  \
           ('sample info' not in expected or
            expected['sample info'] is None):
            # Valid if no sample info is defined at all
            continue
        for key in expected['sample info']:
            if isinstance(expected['sample info'][key],np.ndarray):
                assert np.allclose(sample_info[key],
                                   expected['sample info'][key])
            else:
                assert sample_info[key] == expected['sample info'][key]
                
    
def test_get_wavelength(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        assert np.isclose(expected['wavelength'],data.get_wavelength(cxi))


def test_get_detector_geometry(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        distance, basis, corner = data.get_detector_geometry(cxi)
        assert np.isclose(distance,expected['detector']['distance'])
        assert np.allclose(basis,expected['detector']['basis'])
        if isinstance(expected['detector']['corner'], np.ndarray):
            assert np.allclose(corner, expected['detector']['corner'])
        else:
            assert corner == expected['detector']['corner']

            
def test_get_mask(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        mask = data.get_mask(cxi)
        if expected['mask'] is None and mask is None:
            continue
        assert np.all(data.get_mask(cxi) == expected['mask'])

    
def test_get_data(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        patterns, axes = data.get_data(cxi)
        assert np.allclose(patterns, expected['data'])
        assert axes == expected['axes']


def test_get_ptycho_translations(test_ptycho_cxis):
    for cxi, expected in test_ptycho_cxis:
        assert np.allclose(data.get_ptycho_translations(cxi),
                           expected['translations'])



#
# Then, write a test for the data saving. It should create a .cxi file
# using the data seving tools, and then check that when read with the
# .cxi reading tools that it gets the same things that were written.
#


def test_create_cxi(tmp_path):
    data.create_cxi(tmp_path / 'test_create.cxi')
    with h5py.File(tmp_path / 'test_create.cxi') as f:
        assert f['cxi_version'][()] == 160
        assert 'entry_1' in f

        
def test_add_entry_info(tmp_path):
    entry_info = {'experiment_identifier':'test of cxi file writing tools',
                  'title': 'my cool experiment',
                  'start_time': datetime.datetime.now(),
                  'end_time': datetime.datetime.now()}

    with data.create_cxi(tmp_path / 'test_add_entry_info.cxi') as f:
        data.add_entry_info(f, entry_info)

    with h5py.File(tmp_path / 'test_add_entry_info.cxi') as f:
        read_entry_info = data.get_entry_info(f)

    for key in entry_info:
        if isinstance(entry_info[key], np.ndarray):
            assert np.allclose(entry_info[key], read_entry_info[key])
        else:
            assert entry_info[key] == read_entry_info[key]


def test_add_sample_info(tmp_path):
    sample_info = {'name':'A nice fake sample',
                   'concentration': 10,
                   'mass': 5.3,
                   'temperature': 76,
                   'description': 'A very nice sample',
                   'unit_cell': np.array([1,1,1,90.,90.,90.])}

    with data.create_cxi(tmp_path / 'test_add_sample_info.cxi') as f:
        data.add_sample_info(f, sample_info)

    with h5py.File(tmp_path / 'test_add_sample_info.cxi') as f:
        read_sample_info = data.get_sample_info(f)    

    for key in sample_info:
        if isinstance(sample_info[key], np.ndarray):
            assert np.allclose(sample_info[key], read_sample_info[key])
        elif isinstance(sample_info[key], numbers.Number):
            assert np.isclose(sample_info[key], read_sample_info[key])
        else:
            assert sample_info[key] == read_sample_info[key]
                      

def test_add_source(tmp_path):
    wavelength = 1e-9
    energy = 1.9864459e-25 / wavelength

    with data.create_cxi(tmp_path / 'test_add_source.cxi') as f:
        data.add_source(f, wavelength)

    with h5py.File(tmp_path / 'test_add_source.cxi') as f:
        # Check this directly since we want to make sure it saved
        # the wavelength and energy
        read_wavelength = f['entry_1/instrument_1/source_1/wavelength'][()]
        read_energy = f['entry_1/instrument_1/source_1/energy'][()]

    assert np.isclose( wavelength, read_wavelength)
    assert np.isclose( energy, read_energy)

    
def test_add_detector(tmp_path):
    distance = 0.34
    basis = np.array([[0,-30e-6,0],
                      [-20e-6,0,0]]).astype(np.float32).transpose()
    corner = np.array((2550e-6,3825e-6,0.3)).astype(np.float32)
    
    with data.create_cxi(tmp_path / 'test_add_detector.cxi') as f:
        data.add_detector(f, distance, basis, corner=corner)

    with h5py.File(tmp_path / 'test_add_detector.cxi') as f:
        # Check this directly since we want to make sure it saved
        # the pixel sizes
        d1 = f['entry_1/instrument_1/detector_1']
        read_basis = np.array(d1['basis_vectors'])
        read_x_pix = np.float32(d1['x_pixel_size'])
        read_y_pix = np.float32(d1['y_pixel_size'])
        read_distance = np.float32(d1['distance'])
        read_corner = np.array(d1['corner_position'])

    assert np.isclose(distance, read_distance)
    assert np.allclose(basis, read_basis)
    assert np.isclose(np.linalg.norm(basis[:,1]), read_x_pix)
    assert np.isclose(np.linalg.norm(basis[:,0]), read_y_pix)
    assert np.allclose(corner,read_corner)
   

def test_add_mask(tmp_path):
    mask = (np.random.rand(350,600) > 0.1).astype(np.uint8)

    with data.create_cxi(tmp_path / 'test_add_mask.cxi') as f:
        data.add_mask(f, mask)

    with h5py.File(tmp_path / 'test_add_mask.cxi') as f:
        read_mask = data.get_mask(f)

    assert np.all(mask == read_mask)
    

def test_add_data(tmp_path):
    # First test from numpy, with axes
    fake_data = np.random.rand(100,256,256)
    axes = ['translation','y','x']

    with data.create_cxi(tmp_path / 'test_add_data.cxi') as f:
        data.add_data(f, fake_data, axes)

    with h5py.File(tmp_path / 'test_add_data.cxi') as f:
        # Check this directly since we want to make sure it saved
        # it in all the places it should have
        read_data_1 = np.array(f['entry_1/data_1/data'])
        read_data_2 = np.array(f['entry_1/instrument_1/detector_1/data'])
        read_axes = str(f['entry_1/instrument_1/detector_1/data'].attrs['axes'].decode())
    

    assert np.allclose(fake_data, read_data_1)
    assert np.allclose(fake_data, read_data_2)
    assert 'translation:y:x' == read_axes

    # Then test from torch, without axes
    fake_data = t.from_numpy(fake_data)
    
    with data.create_cxi(tmp_path / 'test_add_data_torch.cxi') as f:
        data.add_data(f, fake_data)

    with h5py.File(tmp_path / 'test_add_data_torch.cxi') as f:    
        read_data, axes = data.get_data(f)

    assert np.allclose(fake_data.numpy(),read_data)


def test_add_ptycho_translations(tmp_path):
    
    translations = np.random.rand(3,100)

    with data.create_cxi(tmp_path / 'test_add_ptycho_translations.cxi') as f:
        data.add_ptycho_translations(f, translations)
    
    with h5py.File(tmp_path / 'test_add_ptycho_translations.cxi') as f:
        # Check this directly since we want to make sure it saved
        # it in all the places it should have
        read_translations_1 = np.array(f['entry_1/data_1/translation'])
        read_translations_2 = np.array(f['entry_1/instrument_1/detector_1/translation'])
        read_translations_3 = np.array(f['entry_1/sample_1/geometry_1/translation'])

    assert np.allclose(-translations.transpose(), read_translations_1)
    assert np.allclose(-translations.transpose(), read_translations_2)
    assert np.allclose(-translations.transpose(), read_translations_3)

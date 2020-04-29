from __future__ import division, print_function, absolute_import
import numpy as np
import h5py
import pytest
import datetime


#
#
# The following few fixtures define some standard data files
# for use to test the data loading capabilities, whether in the
# datasets directly or in the data tools file
#
#

def pytest_addoption(parser):
    parser.addoption(
        "--plot", action="store", default=False, help="plot: True to show test plots"
    )


@pytest.fixture
def show_plot(request):
    return request.config.getoption("--plot")



@pytest.fixture(scope='module')
def ptycho_cxi_1():
    """Creates an example file for CXI ptychography. This file is defined
    to have everything done as correctly as possible with lots of attributes
    defined. It will return both a dictionary describing what is expected
    to be loaded and a file with the data stored in it.
    """

    expected = {}
    f = h5py.File('ptycho_cxi_1','w',driver='core',backing_store=False)

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
    mask = np.zeros((256,256)).astype(np.int32)
    expected['mask'] = np.ones((256,256)).astype(np.bool)
    d1f.create_dataset('mask',data=mask)

    # Create an initial background
    dark = np.ones((256,256)) * 0.01
    expected['dark'] = dark
    d1f.create_dataset('data_dark', data=dark)

    data1f = e1f.create_group('data_1')

    data = np.random.rand(100,256,256).astype(np.float32)
    expected['data'] = data
    d1f.create_dataset('data',data=data)
    data1f['data'] = h5py.SoftLink('/entry_1/instrument_1/detector_1/data')

    d1f['data'].attrs['axes'] = np.string_('translation:y:x')
    expected['axes'] = ['translation','y','x']

    g1f = s1f.create_group('geometry_1')
    orientation = np.array([1.,0,0,0,1,0])
    g1f.create_dataset('orientation', data=orientation)
    s1e['orientation'] = np.array([[1.,0,0],[0,1,0],[0,0,1]])
    translations = np.arange(300).reshape((100,3)).astype(np.float32)
    g1f.create_dataset('translation',data=translations)    
    data1f['translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
    d1f['translation'] = h5py.SoftLink('/entry_1/sample_1/geometry_1/translation')
    expected['translations'] = -translations

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
    f = h5py.File('ptycho_cxi_2','w',driver='core',backing_store=False)

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

    # Test with a set of dark images
    dark = np.ones((10,256,256)) * 0.01
    expected['dark'] = np.nanmean(dark,axis=0)
    d1f.create_dataset('data_dark', data=dark)


    data1f = e1f.create_group('data_1')

    data = np.random.rand(100,256,256).astype(np.float32)
    expected['data'] = data
    d1f.create_dataset('data',data=data)

    expected['axes'] = None

    g1f = s1f.create_group('geometry_1')
    translations = np.arange(300).reshape((100,3)).astype(np.float32)
    g1f.create_dataset('translation',data=translations)
    expected['translations'] = -translations

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
    f = h5py.File('ptycho_cxi_3','w',driver='core',backing_store=False)

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
    mask = np.ones((256,256)).astype(np.uint32) * 0x00001000
    expected['mask'] = np.ones((256,256)).astype(np.bool)
    d1f.create_dataset('mask',data=mask)
    expected['dark'] = None
    
    data1f = e1f.create_group('data_1')

    data = np.random.rand(100,256,256).astype(np.float32)
    expected['data'] = data
    data1f.create_dataset('data',data=data)

    data1f['data'].attrs['axes'] = np.string_('translation:y:x')
    expected['axes'] = ['translation','y','x']

    translations = np.arange(300).reshape((100,3)).astype(np.float32)
    data1f.create_dataset('translation',data=translations)
    expected['translations'] = -translations

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

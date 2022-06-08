Installation
============

Step 1: Download
----------------

The source code for CDTools is hosted on it's `MIT github page`_. Access to the MIT github page can be granted to any member of the MIT community - if you are not a member, access can be arranged through a guest account on MIT's GitHub enterprise server.

.. _`MIT github page`: https://github.mit.edu/Scattering/cdtools

It is recommended that you clone the repository, rather than just downloading the contents, as it remains under development. Cloning the repository will allow you to easily get access to new updates, bug fixes, etc.

Step 2: Install Dependencies
----------------------------

The major dependency for CDTools is pytorch, and because the details of the installation can vary depending on platform, GPU availability, etc, it is recommended that you follow the install instructions on `the pytorch site`_ to install pytorch before the remaining dependencies.

.. _`the pytorch site`: https://pytorch.org/get-started/locally/

If you manage your environment with conda, the remaining dependencies can be installe by running the following command in the top level directory of the package:

.. code:: bash
	  
   $ conda install --file conda_requirements.txt -c conda-forge

This will install all dependencies, including optional dependencies for the tests and docs. For convenience, the full set of dependencies are noted below:
   
CDTools depends on the following packages:

   * `numpy <http://www.numpy.org>`_
   * `scipy <http://www.scipy.org>`_
   * `matplotlib <https://matplotlib.org>`_
   * `pytorch <https://pytorch.org>`_
   * `python-dateutil <https://github.com/dateutil/dateutil/>`_
   * `h5py <https://www.h5py.org/>`_

And has optional dependencies on

   * `pytest <https://docs.pytest.org/>`_
   * `sphinx <https://www.sphinx-doc.org/>`_
   * `sphinx-argparse <https://sphinx-argparse.readthedocs.io>`_
   * `sphinx_rtd_theme <https://sphinx-rtd-theme.readthedocs.io/en/stable/>`_

Finally, the optional depencency on pytest enables the tests to be run to confirm a successful installation. Sphinx and sphinx-argparse are only required if you plan on building the documentation.

Finally, CDTools is NOT python 2 compatible.

Step 3: Install
---------------

To install in CDTools in developer mode (recommended, to allow any updates to be pushed immediately through your system), run the following command from the top level directory (the directory including the setup.py file)

.. code:: bash
	  
   $ pip install -e . --no-deps

To install normally (not in developer mode), run:

.. code:: bash
	  
   $ pip install . --no-deps

  
Step 4: Run The Tests
---------------------

To ensure that the installation has worked correctly, it is recommended that you run the unit tests. After ensuring that pytest is installed, run the following command from the top level directory:

.. code:: bash

   $ pytest


If any tests fail, make sure that you have all the noted dependencies properly installed, and particularly that pytorch was installed with FFT support. If so, `shoot me an email <alevitan@mit.edu>`_ or open an issue on the github page and we'll get to the bottom of it.

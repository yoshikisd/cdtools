Installation
============

Step 1: Download
----------------

The source code for CDTools is hosted on it's `MIT github page`_. Access to the MIT github page can be granted to any member of the MIT community. If you are not a member, access can also be arranged through a `private github tracking repository`_, by contacting `Abe Levitan <alevitan@mit.edu>`_.

.. _`MIT github page`: https://github.mit.edu/Scattering/cdtools
.. _`private github tracking repository`: https://github.com/allevitan/cdtools

It is recommended that you clone the repository, rather than just downloading the contents, as it remains under active development. Cloning the repository will allow you to easily get access to new updates, bug fixes, etc.

Step 2: Install Dependencies
----------------------------

CDTools requires python 3.7 or greater.

The major dependency for CDTools is pytorch (version 1.9.0 or greater). Because the details of the installation can vary depending on platform, GPU availability, etc, it is recommended that you follow the install instructions on `the pytorch site`_ to install pytorch before installing the remaining dependencies.

.. _`the pytorch site`: https://pytorch.org/get-started/locally/

If you manage your environment with conda, the remaining dependencies can be installed by running the following command in the top level directory of the package:

.. code:: bash
	  
   $ conda install --file requirements.txt -c conda-forge

This will install all required dependencies *except for pytorch*, as well as several optional dependencies which are used for the tests and documentation. The full set of dependencies are noted below.
   
Required dependencies:

   * `numpy <http://www.numpy.org>`_ >= 1.0
   * `scipy <http://www.scipy.org>`_ >= 1.0
   * `matplotlib <https://matplotlib.org>`_ >= 2.0
   * `pytorch <https://pytorch.org>`_ >= 1.9.0
   * `python-dateutil <https://github.com/dateutil/dateutil/>`_
   * `h5py <https://www.h5py.org/>`_ >= 2.1

Optional dependencies:

   * `pytest <https://docs.pytest.org/>`_
   * `pooch <https://www.fatiando.org/pooch/latest/>`_
   * `sphinx <https://www.sphinx-doc.org/>`_ >= 4.3.0
   * `sphinx-argparse <https://sphinx-argparse.readthedocs.io>`_
   * `sphinx_rtd_theme <https://sphinx-rtd-theme.readthedocs.io/en/stable/>`_ >= 0.5.1

The file "example_environment.yml", included in the repository's top level directory, contains an example of an environment with all dependencies properly installed on a linux machine with a GPU, circa early 2024.

Step 3: Install
---------------

To install CDTools, run the following command from the top level directory (the directory including the setup.py file).

.. code:: bash
	  
   $ pip install . --no-deps

   
This will install a copy of the code, as it exists at the moment of installation. If you would prefer for changes to the code to propagate to the installed version without reinstalling, install the package in developer mode:

.. code:: bash
	  
   $ pip install -e . --no-deps
   
  
Step 4: Run The Tests
---------------------

To ensure that the installation has worked correctly, it is recommended that you run the unit tests. After ensuring that pytest is installed, run the following command from the top level directory:

.. code:: bash

   $ pytest


If any tests fail, make sure that you have all the noted dependencies properly installed. If you do, and things still aren't working, `send me (Abe Levitan) an email <alevitan@mit.edu>`_  and we'll get to the bottom of it.

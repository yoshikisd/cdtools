""" This module contains various packages of tools supporting common needs

It at times feels a bit silly to keep all the tools siloed under a particular
subpackage such as "projectors" or "measurements", especially given that
"flat is better than nested", but I find something comforting about the
organization and the ability to only pull in the particular set of tools
that one needs for a specific application.

The submodules are all structured as modules with their own __init__ files,
which use an import * statement to import from a file defining the various
functions. This is done to prevent leakage of imported packages into the
namespace of CDTools. I know, we're all consenting adults, but I just hate
having numpy and torch defined under CDTools.tools.cmath, you know?

"""

from __future__ import division, print_function, absolute_import


from CDTools.tools import losses
from CDTools.tools import data
from CDTools.tools import image_processing
from CDTools.tools import initializers
from CDTools.tools import plotting
from CDTools.tools import interactions
from CDTools.tools import propagators
from CDTools.tools import measurements
from CDTools.tools import analysis
from CDTools.tools import atoms

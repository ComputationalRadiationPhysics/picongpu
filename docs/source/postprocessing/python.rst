.. _pp-python:

Python
======

.. sectionauthor:: Axel Huebl, Klaus Steiniger

If you are new to python, get your hands on the tutorials of the following important libraries to get started.

- https://www.python.org/about/gettingstarted/
- https://docs.python.org/3/tutorial/index.html

An easy way to get a Python environment for analysis of PIConGPU data up and running is to download and install
Micromamba

https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

and set up a Micromamba environment with the help of this Micromamba environment file

https://gist.github.com/steindev/d19263d41b0964bcecdfb1f47e18a86e

(see documentation within the file).


Numpy
-----

Numpy is the universal swiss army knife for working on ND arrays in python.

https://docs.scipy.org/doc/numpy-dev/user/quickstart.html


Matplotlib
----------

One common way to visualize plots:

- http://matplotlib.org/faq/usage_faq.html#usage
- https://gist.github.com/ax3l/fc123cb94f59d440f952


Jupyter
-------

Access, share, modify, run and interact with your python scripts from your browser:

https://jupyter.readthedocs.io


openPMD-viewer
--------------

An exploratory framework that visualizes and analyzes data in our HDF5 files thanks to their :ref:`openPMD markup <pp-openPMD>`.
Automatically converts units to SI, interprets iteration steps as time series, annotates axes and provides some domain specific analysis, e.g. for LWFA.
Also provides an interactive GUI for fast exploration via Jupyter notebooks.

* `Project Homepage <https://github.com/openPMD/openPMD-viewer>`_
* `Tutorial <https://github.com/openPMD/openPMD-viewer/tree/master/tutorials>`_


openPMD-api
-----------

A data library that reads (and writes) data in our openPMD files (ADIOS2 and HDF5) to and from Numpy data structures.
Provides an API to correctly convert units to SI, interprets iteration steps correctly, etc.

* `Manual <https://openpmd-api.readthedocs.io/>`_
* `Examples <https://github.com/openPMD/openPMD-api/tree/dev/examples>`_


yt-project
----------

With yt 3.4 or newer, our HDF5 output, which uses the :ref:`openPMD markup <pp-openPMD>`, can be read, processed and visualized with yt.

* `Project Homepage <http://yt-project.org>`_
* `Data Loading <http://yt-project.org/doc/examining/loading_data.html#openpmd-data>`_
* `Data Tutorial <https://gist.github.com/C0nsultant/5808d5f61b271b8f969d5c09f5ca91dc>`_

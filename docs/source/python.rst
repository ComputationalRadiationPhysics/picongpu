Python
======

If you are new to python, get your hands on the tutorials of the following important libraries to get started.

- https://www.python.org/about/gettingstarted/
- https://docs.python.org/3/tutorial/index.html


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

A library that reads and visualizes data in our HDF5 files.
Provides an API to correctly convert units to SI, interpret iteration steps correctly, annotate axis and much more.
Also provides an interactive GUI for fast exploration via Jupyter notebooks.

https://github.com/openPMD/openPMD-viewer/tree/master/tutorials


yt-project (dev)
----------------

Starting with yt 3.4, our HDF5 output, which uses the openPMD markup, can be read, processed and visualized with yt.

http://yt-project.org/docs/dev/


pyDive (experimental)
---------------------

pyDive provides numpy-style array and file processing on distributed memory systems ("numpy on MPI" for data sets that are much larger than your local RAM).
pyDive is currently not ready to interpret openPMD directly, but can work on generated raw ADIOS and HDF5 files.

https://github.com/ComputationalRadiationPhysics/pyDive#documentation

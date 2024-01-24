.. _testing-structure:

Structure
=========

This document describes the general structure of the test-suite and its dependencies.

General structure
-----------------

Each test can be divided into three parts. The first part contains the theory.
The second part contains the object to be compared, in the case of PIConGPU it is the simulation data.
Ultimately, these only have to be compared.
In order to then clearly present the result obtained to the user, an output is also an advantage.

| ./lib/python/test/testsuite
| ├── Math
| | ├── deviation.py
| | ├── physics.py
| | ├── math.py
| | └── _searchData.py
| ├── Output
| | ├── Log.py 
| | └── Viewer.py
| ├── Reader
| | ├── dataReader.py   
| | ├── jsonReader.py
| | └── paramReader.py
| ├── Template
| | ├── Data.py 
| | └── main.py
| └── _checkData.py
|

In the above overview, all ``__init__.py`` and ``_manager.py`` files have been left out for the sake of clarity.

The subpackage reader reads out the data.
It does not matter whether this data is used for the theory part or data from the simulation.
With this, ``Reader`` takes care of most of the first and second part of a test.
However, not completely. For most tests, the data from the simulation cannot be used without further calculations. 
The ``Math`` subpackage provides some important calculation functions.
It also contains the third part of a test.
The ``deviation.py`` file provides the functions needed for comparison.
Finally, ``Output`` encapsulates all functionalities for illustrating the test results and ``Template`` contains a template for developing a new test case.

Dependencies
------------

There are only dependencies on modules in Python. The following modules are not in the standard library:

- scipy
- numpy
- matplotlib

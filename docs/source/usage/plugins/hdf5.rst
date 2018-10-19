.. _usage-plugins-HDF5:

HDF5
----

Stores simulation data such as fields and particles along with domain information,
conversion units etc. as `HDF5 <http://www.hdfgroup.org/HDF5/>`_ files.
It uses `libSplash <https://github.com/ComputationalRadiationPhysics/libSplash>`_ for writing HDF5 data. 
It is used for post-simulation analysis and for **restarts** of the simulation after a crash or an intended stop. 

What is the format of the created HDF5 files?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We write our fields and particles in an open markup called **openPMD**.
You can investigate your files via a large collection of `tools and frameworks <https://github.com/openPMD/openPMD-projects>`_ or your use the native HDF5 bindings of your `favourite programming language <https://en.wikipedia.org/wiki/Hierarchical_Data_Format#Interfaces>`_.

**Resources for a quick-start:**

* `online tutorial <http://www.openPMD.org>`_
* `example files <https://github.com/openPMD/openPMD-example-datasets>`_
* `written standard <https://github.com/openPMD/openPMD-standard>`_ of the openPMD standard
* `list of projects <https://github.com/openPMD/openPMD-projects>`_ supporting openPMD files

External Dependencies
^^^^^^^^^^^^^^^^^^^^^

The plugin is available as soon as the :ref:`libSplash and HDF5 libraries <install-dependencies>` are compiled in.

.param file
^^^^^^^^^^^

The corresponding ``.param`` file is :ref:`fileOutput.param <usage-params-plugins>`.

One can e.g. disable the output of particles by setting:

.. code-block:: cpp

   /* output all species */
   using FileOutputParticles = VectorAllSpecies;
   /* disable */
   using FileOutputParticles = bmpl::vector0< >;

.cfg file
^^^^^^^^^

You can use ``--hdf5.period`` and ``--hdf5.file`` to specify the output period and path and name of the created fileset.
For example, ``--hdf5.period 128 --hdf5.file simData --hdf5.source 'species_all'`` will write only the particle species data to files of the form ``simData_0.h5``, ``simData_128.h5`` in the default simulation output directory every 128 steps.
Note that this plugin will only be available if libSplash and HDF5 is found during compile configuration.

============================ ====================================================================
PIConGPU command line option Description
============================ ====================================================================
``--hdf5.period``            Period after which simulation data should be stored on disk.
``--hdf5.file``              Relative or absolute fileset prefix for simulation data.
                             If relative, files are stored under ``simOutput/``.
``--hdf5.source``            Select data sources to dump. Default is ``species_all,fields_all``, 
                             which dumps all fields and particle species.
============================ ====================================================================

.. note::

   This plugin is a multi plugin. 
   Command line parameter can be used multiple times to create e.g. dumps with different dumping period.
   In the case where a optional parameter with a default value is explicitly defined the parameter will be always passed to the instance of the multi plugin where the parameter is not set.
   e.g. 

   .. code-block:: bash

      --hdf5.period 128 --hdf5.file simData1 
      --hdf5.period 1000 --hdf5.file simData2 --hdf5.source 'species_all'

   creates two plugins:
 
   #. dump **all species data** each 128th time step.
   #. dump **all fields and species data** (this is the default) data each 1000th time step.

Memory Complexity
^^^^^^^^^^^^^^^^^

Accelerator
"""""""""""

no extra allocations.

Host
""""

During I/O, each complete particle species is allocated one after an other.

Additional Tools
^^^^^^^^^^^^^^^^

See our :ref:`openPMD <pp-openPMD>` chapter.
